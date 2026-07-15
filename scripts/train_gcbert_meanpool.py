"""
GCBERT Mean-Pool + MLP Baseline.
Uses pre-computed graph.x embeddings directly.
No GPU re-encoding needed. Fast.
Tests: can simple pooling + MLP match SiameseGAT?
"""
import torch, torch.nn as nn, torch.nn.functional as F
import json, os, gc, time, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GroupKFold
from collections import defaultdict

class EmbeddingPairDataset(Dataset):
    def __init__(self, graphs_a, graphs_b, labels, swap_aug=False):
        self.labels   = labels
        self.swap_aug = swap_aug
        print("  Pooling embeddings...", flush=True)
        self.emb_a = [g.x[:g.num_code_tokens].mean(0) for g in graphs_a]
        self.emb_b = [g.x[:g.num_code_tokens].mean(0) for g in graphs_b]
        print("  Done.")
    def __len__(self):
        return len(self.labels) * (2 if self.swap_aug else 1)
    def __getitem__(self, idx):
        if self.swap_aug and idx >= len(self.labels):
            i = idx - len(self.labels)
            return self.emb_b[i], self.emb_a[i], \
                   torch.tensor(1 - self.labels[i], dtype=torch.long)
        return self.emb_a[idx], self.emb_b[idx], \
               torch.tensor(self.labels[idx], dtype=torch.long)

def collate(batch):
    a, b, l = zip(*batch)
    return torch.stack(a), torch.stack(b), torch.stack(l)

class SiameseMLP(nn.Module):
    """
    Tests whether mean-pooled GCBERT + learned comparison = SiameseGAT.
    If this matches SiameseGAT F1, graph structure truly adds nothing.
    If this fails, graph structure is necessary.
    """
    def __init__(self, dim=768, hidden=256, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
    def forward(self, a, b):
        feat = torch.cat([a, b, (a-b).abs(), a*b], dim=-1)
        return self.mlp(feat)

def get_groups(metadata):
    groups = [(m.get('problem_id') or (m.get('pair_id','unk').split('_')[1] if len(m.get('pair_id','unk').split('_')) >= 3 else m.get('pair_id','unk'))) for m in metadata]
    uniq = sorted(set(groups))
    gmap = {g:i for i,g in enumerate(uniq)}
    return np.array([gmap[g] for g in groups])

def run(graph_data, args, output_dir, device):
    labels   = np.array(graph_data['labels'])
    metadata = graph_data['metadata']
    groups   = get_groups(metadata)
    
    n_groups = len(set(groups.tolist()))
    if n_groups >= args.n_folds:
        splits = list(GroupKFold(args.n_folds).split(
            labels, labels, groups))
    else:
        splits = list(StratifiedKFold(
            args.n_folds, shuffle=True, random_state=42
        ).split(labels, labels))
    
    print(f"\n{'='*60}")
    print(f"MODEL: gcbert_meanpool_mlp")
    print(f"FOLDS: {args.n_folds}, PAIRS: {len(labels)}")
    print(f"{'='*60}\n")
    
    fold_results = []
    all_preds, all_labels = [], []
    
    for fold, (tr_idx, te_idx) in enumerate(splits):
        print(f"\nFOLD {fold+1}/{args.n_folds} "
              f"Train:{len(tr_idx)} Test:{len(te_idx)}")
        
        tr_ds = EmbeddingPairDataset(
            [graph_data['graph_a'][i] for i in tr_idx],
            [graph_data['graph_b'][i] for i in tr_idx],
            [graph_data['labels'][i]  for i in tr_idx],
            swap_aug=True)
        te_ds = EmbeddingPairDataset(
            [graph_data['graph_a'][i] for i in te_idx],
            [graph_data['graph_b'][i] for i in te_idx],
            [graph_data['labels'][i]  for i in te_idx],
            swap_aug=False)
        
        tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True,
                           collate_fn=collate)
        te_ld = DataLoader(te_ds, batch_size=64, shuffle=False,
                           collate_fn=collate)
        
        model = SiameseMLP().to(device)
        if fold == 0:
            n = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n:,}")
        
        opt  = torch.optim.AdamW(model.parameters(), 
                                  lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        best_f1, best_m, patience = 0, {}, 0
        
        for ep in range(args.max_epochs):
            t0 = time.time()
            model.train()
            tr_loss = tr_correct = tr_total = 0
            for a, b, l in tr_ld:
                a,b,l = a.to(device), b.to(device), l.to(device)
                opt.zero_grad()
                logits = model(a, b)
                loss = crit(logits, l)
                loss.backward(); opt.step()
                tr_loss    += loss.item() * l.size(0)
                tr_correct += (logits.argmax(1)==l).sum().item()
                tr_total   += l.size(0)
            
            model.eval()
            preds, labs, probs = [], [], []
            with torch.no_grad():
                for a, b, l in te_ld:
                    a,b,l = a.to(device), b.to(device), l.to(device)
                    logits = model(a, b)
                    preds.extend(logits.argmax(1).cpu().tolist())
                    labs.extend(l.cpu().tolist())
                    probs.extend(F.softmax(logits,-1)[:,1].cpu().tolist())
            
            f1  = f1_score(labs, preds, average='macro')
            auc = roc_auc_score(labs, probs)
            acc = accuracy_score(labs, preds)
            dt  = time.time() - t0
            
            print(f"  Ep {ep+1:3d}: "
                  f"tr_loss={tr_loss/tr_total:.4f} "
                  f"tr_acc={tr_correct/tr_total:.3f} | "
                  f"te_f1={f1:.3f} te_auc={auc:.3f} [{dt:.1f}s]")
            
            if f1 > best_f1:
                best_f1 = f1
                best_m  = {'f1':f1,'auc':auc,'acc':acc,
                           'preds':preds,'labels':labs}
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    print(f"  Early stop ep {ep+1}")
                    break
        
        print(f"  BEST F1={best_f1:.4f} AUC={best_m['auc']:.4f}")
        fold_results.append({
            'fold':fold,'f1_macro':best_f1,
            'accuracy':best_m['acc'],'auc':best_m['auc']})
        all_preds.extend(best_m['preds'])
        all_labels.extend(best_m['labels'])
        
        # Per-dataset breakdown
        te_meta = [metadata[i] for i in te_idx]
        ds_m = defaultdict(lambda: {'p':[],'l':[]})
        for p,l,m in zip(best_m['preds'],best_m['labels'],te_meta):
            ds_m[m['dataset']]['p'].append(p)
            ds_m[m['dataset']]['l'].append(l)
        for ds,dm in ds_m.items():
            dsf1 = f1_score(dm['l'],dm['p'],average='macro')
            print(f"    {ds}: F1={dsf1:.3f} (n={len(dm['l'])})")
    
    print(f"\n{'='*60}")
    print("AGGREGATE — gcbert_meanpool_mlp")
    for m in ['f1_macro','accuracy','auc']:
        vals = [r[m] for r in fold_results]
        print(f"  {m}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Correct','Buggy'], digits=4)}")
    
    os.makedirs(output_dir, exist_ok=True)
    import json as _json
    with open(os.path.join(output_dir,'results.json'),'w') as f:
        _json.dump({'model':'gcbert_meanpool_mlp',
                    'fold_results':fold_results,
                    'summary':{m:{'mean':float(np.mean([r[m] for r in fold_results])),
                                  'std':float(np.std([r[m] for r in fold_results]))}
                               for m in ['f1_macro','accuracy','auc']}},
                   f, indent=2)
    print(f"Saved to {output_dir}/results.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang',       required=True)
    ap.add_argument('--graph-dir',  default='data/graphs')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--device',     default='cuda:1')
    ap.add_argument('--max-epochs', type=int, default=50)
    ap.add_argument('--patience',   type=int, default=10)
    ap.add_argument('--n-folds',    type=int, default=5)
    args = ap.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f'./outputs/gcbert_meanpool_{args.lang}'
    
    lang = args.lang.lower()
    gdir = args.graph_dir
    graph_data = {'graph_a':[],'graph_b':[],'labels':[],'metadata':[]}
    
    for ds in ['codenet','humanevalfix']:
        fp = os.path.join(gdir, f'graph_data_{ds}_{lang}.pt')
        if not os.path.exists(fp): continue
        print(f"Loading {fp}...", flush=True)
        d = torch.load(fp, weights_only=False, map_location='cpu')
        graph_data['graph_a'].extend(d['graph_a'])
        graph_data['graph_b'].extend(d['graph_b'])
        graph_data['labels'].extend(d['labels'])
        graph_data['metadata'].extend(d['metadata'])
        del d; gc.collect()
    
    print(f"Total: {len(graph_data['labels'])} pairs")
    run(graph_data, args, args.output_dir, args.device)

if __name__ == '__main__':
    main()
