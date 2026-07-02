"""
Specification-Aware Siamese GAT for Code Correctness Verification.

Extends model.py with IO specification graph encoding:
- SpecAwareSiameseGAT: Fuses code pair embeddings with spec embeddings
- CrossAttentionFusion: Optional cross-modal attention (spec ↔ code nodes)
- Full ablation support via config flags

Architecture:
    Code_A (graph) → GAT_code → pool → z_a ─┐
    Code_B (graph) → GAT_code → pool → z_b ─┤→ Fusion → MLP → ŷ
    Spec   (graph) → GAT_spec → pool → z_s ─┘

Fusion modes:
    "concat":  [z_a; z_b; z_a-z_b; z_a*z_b; z_s; z_a-z_s; z_b-z_s]
    "gate":    learned gating of spec contribution
    "cross_attention": node-level cross-attention before pooling

Usage:
    model = SpecAwareSiameseGAT(config)
    result = model(data_a, data_b, data_spec)       # spec-aware
    result = model(data_a, data_b, data_spec=None)   # fallback (ablation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax as pyg_softmax
from typing import Optional, Tuple, Dict, List

from model import GATEncoder, AttentionPooling, count_parameters


# ═══════════════════════════════════════
# CROSS-ATTENTION FUSION MODULE
# ═══════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Cross-modal attention between spec and code node embeddings.
    
    After GAT encoding but BEFORE pooling:
    - Code nodes attend to spec nodes → spec-informed code representation
    - Spec nodes attend to code nodes → code-informed spec representation
    
    This allows fine-grained alignment: which spec tokens (IO values)
    correspond to which code tokens (variables, operations).
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.code_to_spec = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spec_to_code = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_code = nn.LayerNorm(hidden_dim)
        self.norm_spec = nn.LayerNorm(hidden_dim)
        self.ffn_code = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_spec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_code2 = nn.LayerNorm(hidden_dim)
        self.norm_spec2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def _to_dense_batch(self, x: torch.Tensor, batch: torch.Tensor):
        """Convert PyG sparse node tensors to dense [B, max_N, D] for MHA."""
        from torch_geometric.utils import to_dense_batch
        dense, mask = to_dense_batch(x, batch)  # [B, max_N, D], [B, max_N]
        return dense, mask
    
    def _from_dense_batch(self, dense: torch.Tensor, mask: torch.Tensor,
                          batch: torch.Tensor):
        """Convert dense back to sparse PyG format."""
        # Gather only the valid (non-padded) nodes
        batch_size, max_n, dim = dense.shape
        nodes = []
        for b in range(batch_size):
            n = mask[b].sum().item()
            nodes.append(dense[b, :n])
        return torch.cat(nodes, dim=0)
    
    def forward(self, code_nodes: torch.Tensor, spec_nodes: torch.Tensor,
                code_batch: torch.Tensor, spec_batch: torch.Tensor):
        """
        Args:
            code_nodes: [N_code, D] node embeddings from code GAT
            spec_nodes: [N_spec, D] node embeddings from spec GAT
            code_batch: [N_code] batch indices
            spec_batch: [N_spec] batch indices
        Returns:
            enhanced_code: [N_code, D] spec-informed code nodes
            enhanced_spec: [N_spec, D] code-informed spec nodes
        """
        code_dense, code_mask = self._to_dense_batch(code_nodes, code_batch)
        spec_dense, spec_mask = self._to_dense_batch(spec_nodes, spec_batch)
        
        # Invert masks for MHA (True = ignore in key_padding_mask)
        code_pad = ~code_mask
        spec_pad = ~spec_mask
        
        # Code attends to Spec: "What does the spec say about this code?"
        code_attn_out, _ = self.code_to_spec(
            query=code_dense, key=spec_dense, value=spec_dense,
            key_padding_mask=spec_pad
        )
        code_enhanced = self.norm_code(code_dense + self.dropout(code_attn_out))
        code_enhanced = self.norm_code2(code_enhanced + self.ffn_code(code_enhanced))
        
        # Spec attends to Code: "What does the code do relative to this spec?"
        spec_attn_out, _ = self.spec_to_code(
            query=spec_dense, key=code_dense, value=code_dense,
            key_padding_mask=code_pad
        )
        spec_enhanced = self.norm_spec(spec_dense + self.dropout(spec_attn_out))
        spec_enhanced = self.norm_spec2(spec_enhanced + self.ffn_spec(spec_enhanced))
        
        # Back to sparse PyG format
        enhanced_code = self._from_dense_batch(code_enhanced, code_mask, code_batch)
        enhanced_spec = self._from_dense_batch(spec_enhanced, spec_mask, spec_batch)
        
        return enhanced_code, enhanced_spec


# ═══════════════════════════════════════
# GATED SPEC FUSION
# ═══════════════════════════════════════

class GatedSpecFusion(nn.Module):
    """
    Learned gating to control how much spec information flows into classification.
    
    gate = σ(W_g · [z_code; z_spec])
    z_fused = gate * z_spec + (1 - gate) * z_code_only_features
    
    This allows the model to learn when spec is helpful vs noisy.
    """
    
    def __init__(self, code_dim: int, spec_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(code_dim + spec_dim, code_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(spec_dim, code_dim)
    
    def forward(self, z_code_features: torch.Tensor, z_spec: torch.Tensor):
        """
        Args:
            z_code_features: [B, code_dim] - the code comparison features
            z_spec: [B, spec_dim] - spec embedding
        Returns:
            fused: [B, code_dim] - gated fusion
        """
        z_spec_proj = self.transform(z_spec)
        g = self.gate(torch.cat([z_code_features, z_spec], dim=-1))
        return z_code_features + g * z_spec_proj


# ═══════════════════════════════════════
# SPEC-AWARE SIAMESE GAT
# ═══════════════════════════════════════

class SpecAwareSiameseGAT(nn.Module):
    """
    Siamese GAT with specification graph awareness.
    
    Three fusion modes:
    1. "concat" (default): Concatenate spec features with code comparison vector
       [z_a; z_b; z_a-z_b; z_a*z_b; z_s; z_a-z_s; z_b-z_s] → MLP
       
    2. "gate": Learned gating of spec contribution
       code_features + gate * spec_features → MLP
       
    3. "cross_attention": Node-level cross-attention before pooling
       code_nodes ←→ spec_nodes → pool → fusion → MLP
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        hidden_dim = config.get("hidden_dim", 256)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.3)
        attn_dropout = config.get("attention_dropout", 0.1)
        mlp_hidden = config.get("mlp_hidden", 256)
        mlp_dropout = config.get("mlp_dropout", 0.3)
        num_classes = config.get("num_classes", 2)
        pooling_type = config.get("pooling", "attention")
        input_dim = config.get("embedding_dim", 768)
        fusion_mode = config.get("fusion_mode", "concat")
        shared_encoder = config.get("shared_spec_encoder", False)
        use_cross_attn = config.get("cross_attention", False)
        
        self.fusion_mode = fusion_mode
        self.use_cross_attn = use_cross_attn
        self.hidden_dim = hidden_dim
        
        # ── Code Encoder (shared for both code_a and code_b) ──
        self.code_encoder = GATEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            dropout=dropout, attention_dropout=attn_dropout,
        )
        
        # ── Spec Encoder ──
        if shared_encoder:
            self.spec_encoder = self.code_encoder
        else:
            self.spec_encoder = GATEncoder(
                input_dim=input_dim, hidden_dim=hidden_dim,
                num_heads=num_heads, num_layers=num_layers,
                dropout=dropout, attention_dropout=attn_dropout,
            )
        
        # ── Pooling ──
        if pooling_type == "attention":
            self.code_pool = AttentionPooling(hidden_dim)
            self.spec_pool = AttentionPooling(hidden_dim) if not shared_encoder else self.code_pool
        else:
            self.code_pool = None
            self.spec_pool = None
        self.pooling_type = pooling_type
        
        # ── Cross-Attention (optional) ──
        if use_cross_attn:
            self.cross_attn = CrossAttentionFusion(
                hidden_dim, num_heads=num_heads, dropout=attn_dropout
            )
        
        # ── Fusion & Classifier ──
        if fusion_mode == "concat":
            # [z_a; z_b; z_a-z_b; z_a*z_b; z_s; z_a-z_s; z_b-z_s] = 7 * hidden_dim
            classifier_input = hidden_dim * 7
        elif fusion_mode == "gate":
            # Gated: same size as code-only (4 * hidden_dim) after gating
            self.spec_gate = GatedSpecFusion(hidden_dim * 4, hidden_dim)
            classifier_input = hidden_dim * 4
        else:
            classifier_input = hidden_dim * 7  # fallback to concat
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden // 2, num_classes),
        )
    
    def _pool(self, h: torch.Tensor, batch: torch.Tensor,
              pool_module: Optional[AttentionPooling]) -> torch.Tensor:
        """Pool node embeddings to graph-level."""
        if self.pooling_type == "attention" and pool_module is not None:
            return pool_module(h, batch)
        elif self.pooling_type == "max":
            return global_max_pool(h, batch)
        else:
            return global_mean_pool(h, batch)
    
    def encode_code(self, data: Data, return_attention: bool = False):
        """Encode a code graph."""
        h, attn = self.code_encoder(data.x, data.edge_index, return_attention)
        z = self._pool(h, data.batch, self.code_pool)
        return z, h, attn
    
    def encode_spec(self, data: Data, return_attention: bool = False):
        """Encode a spec graph."""
        h, attn = self.spec_encoder(data.x, data.edge_index, return_attention)
        z = self._pool(h, data.batch, self.spec_pool)
        return z, h, attn
    
    def forward(self, data_a: Data, data_b: Data,
                data_spec: Optional[Data] = None,
                return_attention: bool = False) -> Dict:
        """
        Forward pass.
        
        Args:
            data_a: Batch of code graphs (version A / correct)
            data_b: Batch of code graphs (version B / buggy)
            data_spec: Batch of spec graphs (IO examples). None for ablation.
            return_attention: Return attention weights for localization
        
        Returns:
            Dict with logits, embeddings, optionally attention weights
        """
        # ── Encode code graphs (shared weights) ──
        z_a, h_a, attn_a = self.encode_code(data_a, return_attention)
        z_b, h_b, attn_b = self.encode_code(data_b, return_attention)
        
        # ── Encode spec graph ──
        has_spec = data_spec is not None
        if has_spec:
            if self.use_cross_attn:
                # Get node-level spec embeddings
                h_s_raw, _ = self.spec_encoder(data_spec.x, data_spec.edge_index, False)
                
                # Cross-attention: code_a ↔ spec
                h_a_enhanced, h_s_a = self.cross_attn(h_a, h_s_raw, data_a.batch, data_spec.batch)
                z_a_ca = self._pool(h_a_enhanced, data_a.batch, self.code_pool)
                
                # Cross-attention: code_b ↔ spec
                h_b_enhanced, h_s_b = self.cross_attn(h_b, h_s_raw, data_b.batch, data_spec.batch)
                z_b_ca = self._pool(h_b_enhanced, data_b.batch, self.code_pool)
                
                z_s = self._pool(h_s_raw, data_spec.batch, self.spec_pool)
                
                # Use cross-attention-enhanced code embeddings
                z_a_final = z_a_ca
                z_b_final = z_b_ca
            else:
                z_s, h_s, attn_s = self.encode_spec(data_spec, return_attention)
                z_a_final = z_a
                z_b_final = z_b
        else:
            z_s = torch.zeros_like(z_a)
            z_a_final = z_a
            z_b_final = z_b
        
        # ── Fusion ──
        diff = z_a_final - z_b_final
        product = z_a_final * z_b_final
        
        if self.fusion_mode == "concat":
            spec_code_a = z_a_final - z_s  # Distance: code_a to spec
            spec_code_b = z_b_final - z_s  # Distance: code_b to spec
            combined = torch.cat([
                z_a_final, z_b_final, diff, product,
                z_s, spec_code_a, spec_code_b
            ], dim=-1)
        
        elif self.fusion_mode == "gate":
            code_features = torch.cat([z_a_final, z_b_final, diff, product], dim=-1)
            combined = self.spec_gate(code_features, z_s)
        
        else:
            # Default to concat
            spec_code_a = z_a_final - z_s
            spec_code_b = z_b_final - z_s
            combined = torch.cat([
                z_a_final, z_b_final, diff, product,
                z_s, spec_code_a, spec_code_b
            ], dim=-1)
        
        logits = self.classifier(combined)
        
        # ── Build result dict ──
        result = {
            "logits": logits,
            "z_a": z_a,
            "z_b": z_b,
            "z_s": z_s if has_spec else None,
        }
        
        if return_attention:
            result["attn_a"] = attn_a
            result["attn_b"] = attn_b
            result["h_a"] = h_a
            result["h_b"] = h_b
            if has_spec:
                result["h_s"] = h_s_raw if self.use_cross_attn else h_s
            if self.pooling_type == "attention" and self.code_pool is not None:
                result["pool_attn_a"] = self.code_pool.get_attention_weights(h_a, data_a.batch)
                result["pool_attn_b"] = self.code_pool.get_attention_weights(h_b, data_b.batch)
        
        return result


# ═══════════════════════════════════════
# CODE-ONLY WRAPPER (for fair comparison)
# ═══════════════════════════════════════

class SpecAwareSiameseGAT_CodeOnly(SpecAwareSiameseGAT):
    """
    Wrapper that always passes data_spec=None.
    Used for ablation: "same architecture, no spec input."
    This isolates the effect of spec information vs architecture changes.
    """
    def forward(self, data_a, data_b, data_spec=None, **kwargs):
        return super().forward(data_a, data_b, data_spec=None, **kwargs)


# ═══════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════

def build_model(config: Dict) -> nn.Module:
    """
    Build model based on config.
    
    config["model_type"]:
        "siamese_gat"           → Original SiameseGAT (from model.py)
        "spec_siamese_gat"      → SpecAwareSiameseGAT
        "spec_code_only"        → SpecAwareSiameseGAT with spec=None (ablation)
    """
    from model import SiameseGAT
    
    model_type = config.get("model_type", "siamese_gat")
    
    if model_type == "siamese_gat":
        return SiameseGAT(config)
    elif model_type == "spec_siamese_gat":
        return SpecAwareSiameseGAT(config)
    elif model_type == "spec_code_only":
        return SpecAwareSiameseGAT_CodeOnly(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ═══════════════════════════════════════
# TESTS
# ═══════════════════════════════════════

if __name__ == "__main__":
    config = {
        "hidden_dim": 256, "num_heads": 4, "num_layers": 2,
        "dropout": 0.3, "attention_dropout": 0.1,
        "mlp_hidden": 256, "mlp_dropout": 0.3,
        "num_classes": 2, "pooling": "attention", "embedding_dim": 768,
        "fusion_mode": "concat", "shared_spec_encoder": False,
        "cross_attention": False,
    }
    
    # Create dummy data
    def make_graph(n_nodes=50):
        return Data(
            x=torch.randn(n_nodes, 768),
            edge_index=torch.randint(0, n_nodes, (2, 200)),
            batch=torch.zeros(n_nodes, dtype=torch.long),
        )
    
    # Test 1: Concat fusion
    print("Test 1: Concat fusion (no cross-attention)")
    model = SpecAwareSiameseGAT(config)
    params = count_parameters(model)
    print(f"  Parameters: {params['trainable_human']}")
    
    data_a, data_b = make_graph(), make_graph()
    data_spec = make_graph(30)  # Spec graphs are smaller
    
    with torch.no_grad():
        result = model(data_a, data_b, data_spec)
    print(f"  Logits: {result['logits'].shape}")
    print(f"  z_s: {result['z_s'].shape}")
    
    # Test 2: Without spec (ablation)
    with torch.no_grad():
        result_no_spec = model(data_a, data_b, data_spec=None)
    print(f"  No-spec logits: {result_no_spec['logits'].shape}")
    print(f"  z_s is None: {result_no_spec['z_s'] is None}")
    
    # Test 3: Cross-attention
    print("\nTest 2: Cross-attention fusion")
    config["cross_attention"] = True
    model_ca = SpecAwareSiameseGAT(config)
    params_ca = count_parameters(model_ca)
    print(f"  Parameters: {params_ca['trainable_human']}")
    
    with torch.no_grad():
        result_ca = model_ca(data_a, data_b, data_spec)
    print(f"  Logits: {result_ca['logits'].shape}")
    
    # Test 4: Gated fusion
    print("\nTest 3: Gated fusion")
    config["cross_attention"] = False
    config["fusion_mode"] = "gate"
    model_gate = SpecAwareSiameseGAT(config)
    params_gate = count_parameters(model_gate)
    print(f"  Parameters: {params_gate['trainable_human']}")
    
    with torch.no_grad():
        result_gate = model_gate(data_a, data_b, data_spec)
    print(f"  Logits: {result_gate['logits'].shape}")
    
    # Test 5: Attention for localization
    print("\nTest 4: Return attention weights")
    config["fusion_mode"] = "concat"
    model_attn = SpecAwareSiameseGAT(config)
    with torch.no_grad():
        result_attn = model_attn(data_a, data_b, data_spec, return_attention=True)
    print(f"  pool_attn_a: {result_attn['pool_attn_a'].shape}")
    print(f"  Has h_s: {'h_s' in result_attn}")
    
    # Test 6: Factory
    print("\nTest 5: Factory builds")
    for mt in ["siamese_gat", "spec_siamese_gat", "spec_code_only"]:
        config["model_type"] = mt
        m = build_model(config)
        print(f"  {mt}: {type(m).__name__}")
    
    print("\n✓ All tests passed!")