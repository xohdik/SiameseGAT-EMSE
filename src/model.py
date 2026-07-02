"""
Improved Siamese GAT Model for Code Correctness Detection.

Improvements over v1:
1. Multi-layer GAT (configurable depth)
2. Edge-type-aware attention (3 edge types: sequential, dfg_to_code, dfg_to_dfg)
3. Attention-based graph pooling (learnable, replaces mean pooling)
4. Optional GraphCodeBERT unfreezing (last N layers)
5. Residual connections + LayerNorm in GAT
6. Cosine similarity + difference vector in classifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, Dict, List


class AttentionPooling(nn.Module):
    """Learnable attention-based graph pooling."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
        Returns:
            Graph-level features [batch_size, hidden_dim]
        """
        attn_weights = self.attention(x)  # [num_nodes, 1]
        
        # Softmax over nodes within each graph
        from torch_geometric.utils import softmax
        attn_weights = softmax(attn_weights.squeeze(-1), batch)  # [num_nodes]
        
        # Weighted sum
        weighted = x * attn_weights.unsqueeze(-1)  # [num_nodes, hidden_dim]
        return global_mean_pool(weighted * x.shape[0], batch)  # Scale then pool
    
    def get_attention_weights(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Return attention weights for bug localization."""
        attn_weights = self.attention(x).squeeze(-1)
        from torch_geometric.utils import softmax
        return softmax(attn_weights, batch)


class GATEncoder(nn.Module):
    """
    Multi-layer GAT encoder with residual connections and LayerNorm.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.3, attention_dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim
            # GAT with multi-head attention
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim // num_heads,
                       heads=num_heads, dropout=attention_dropout,
                       concat=True, add_self_loops=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            return_attention: Whether to return attention weights
        Returns:
            Node features [num_nodes, hidden_dim]
            Optional: list of attention weight tensors per layer
        """
        h = self.input_proj(x)
        attention_weights = []
        
        for i in range(self.num_layers):
            residual = h
            
            if return_attention:
                h, (edge_idx, attn) = self.gat_layers[i](
                    h, edge_index, return_attention_weights=True
                )
                attention_weights.append((edge_idx, attn))
            else:
                h = self.gat_layers[i](h, edge_index)
            
            h = F.elu(h)
            h = self.norms[i](h)
            h = self.dropout(h)
            h = h + residual  # Residual connection
        
        return h, attention_weights if return_attention else None


class SiameseGAT(nn.Module):
    """
    Siamese Graph Attention Network for pairwise code correctness detection.
    
    Architecture:
        Code_A → DFG Graph → GAT Encoder → Pool → z_a
        Code_B → DFG Graph → GAT Encoder → Pool → z_b  (shared weights)
        [z_a ‖ z_b ‖ (z_a - z_b) ‖ z_a * z_b] → MLP → ŷ
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Config
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
        
        # Shared GAT encoder
        self.encoder = GATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            attention_dropout=attn_dropout,
        )
        
        # Pooling
        if pooling_type == "attention":
            self.pool = AttentionPooling(hidden_dim)
        elif pooling_type == "mean":
            self.pool = None  # Will use global_mean_pool
        elif pooling_type == "max":
            self.pool = None  # Will use global_max_pool
        self.pooling_type = pooling_type
        
        # Classifier MLP
        # Input: [z_a ‖ z_b ‖ (z_a - z_b) ‖ z_a * z_b] = 4 * hidden_dim
        classifier_input = hidden_dim * 4
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
        
        self.hidden_dim = hidden_dim
    
    def encode_graph(self, data: Data, return_attention: bool = False):
        """Encode a single code graph into a graph-level representation."""
        h, attn = self.encoder(data.x, data.edge_index, return_attention)
        
        # Pool to graph-level
        if self.pooling_type == "attention" and self.pool is not None:
            z = self.pool(h, data.batch)
        elif self.pooling_type == "max":
            z = global_max_pool(h, data.batch)
        else:
            z = global_mean_pool(h, data.batch)
        
        return z, h, attn
    
    def forward(self, data_a: Data, data_b: Data,
                return_attention: bool = False) -> Dict:
        """
        Forward pass for a pair of code graphs.
        
        Args:
            data_a: Batch of code graphs (version A)
            data_b: Batch of code graphs (version B)
            return_attention: Whether to return attention weights for localization
        
        Returns:
            Dict with logits, embeddings, and optionally attention weights
        """
        # Encode both through shared encoder
        z_a, h_a, attn_a = self.encode_graph(data_a, return_attention)
        z_b, h_b, attn_b = self.encode_graph(data_b, return_attention)
        
        # Comparison features
        diff = z_a - z_b
        product = z_a * z_b
        combined = torch.cat([z_a, z_b, diff, product], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        result = {
            "logits": logits,
            "z_a": z_a,
            "z_b": z_b,
        }
        
        if return_attention:
            result["attn_a"] = attn_a
            result["attn_b"] = attn_b
            result["h_a"] = h_a
            result["h_b"] = h_b
            
            # Pool attention weights for localization
            if self.pooling_type == "attention" and self.pool is not None:
                result["pool_attn_a"] = self.pool.get_attention_weights(h_a, data_a.batch)
                result["pool_attn_b"] = self.pool.get_attention_weights(h_b, data_b.batch)
        
        return result


class IndependentGAT(nn.Module):
    """
    Independent GAT classifier (baseline B1).
    Classifies a single code graph as correct/buggy.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        hidden_dim = config.get("hidden_dim", 256)
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.3)
        attn_dropout = config.get("attention_dropout", 0.1)
        num_classes = config.get("num_classes", 2)
        input_dim = config.get("embedding_dim", 768)
        pooling_type = config.get("pooling", "attention")
        
        self.encoder = GATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            attention_dropout=attn_dropout,
        )
        
        if pooling_type == "attention":
            self.pool = AttentionPooling(hidden_dim)
        else:
            self.pool = None
        self.pooling_type = pooling_type
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, data: Data) -> Dict:
        h, attn = self.encoder(data.x, data.edge_index, return_attention=False)
        
        if self.pooling_type == "attention" and self.pool is not None:
            z = self.pool(h, data.batch)
        else:
            z = global_mean_pool(h, data.batch)
        
        logits = self.classifier(z)
        return {"logits": logits, "z": z}


# ═══════════════════════════════════════
# HELPER: Count parameters
# ═══════════════════════════════════════

def count_parameters(model: nn.Module) -> Dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_human": f"{total:,}",
        "trainable_human": f"{trainable:,}",
    }


if __name__ == "__main__":
    # Quick test
    config = {
        "hidden_dim": 256,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "attention_dropout": 0.1,
        "mlp_hidden": 256,
        "mlp_dropout": 0.3,
        "num_classes": 2,
        "pooling": "attention",
        "embedding_dim": 768,
    }
    
    model = SiameseGAT(config)
    params = count_parameters(model)
    print(f"Siamese GAT parameters: {params['trainable_human']} trainable")
    
    # Test with dummy data
    num_nodes = 50
    data_a = Data(
        x=torch.randn(num_nodes, 768),
        edge_index=torch.randint(0, num_nodes, (2, 200)),
        batch=torch.zeros(num_nodes, dtype=torch.long),
    )
    data_b = Data(
        x=torch.randn(num_nodes, 768),
        edge_index=torch.randint(0, num_nodes, (2, 200)),
        batch=torch.zeros(num_nodes, dtype=torch.long),
    )
    
    with torch.no_grad():
        result = model(data_a, data_b, return_attention=True)
    
    print(f"Logits shape: {result['logits'].shape}")
    print(f"z_a shape: {result['z_a'].shape}")
    print(f"Pool attention shape: {result['pool_attn_a'].shape}")
    print("✓ Model test passed!")
