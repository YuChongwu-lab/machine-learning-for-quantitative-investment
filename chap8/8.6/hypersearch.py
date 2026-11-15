"""Hyperparameter search over the data-construction parameters corr_window and
corr_threshold, scored by average Spearman-IC."""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from data_utils import create_enhanced_dataset
from metrics import _compute_signal, _spearman_corr


class HyperparameterSearchEngine:
    def __init__(self, model_class, search_space: Dict[str, List],
                 search_method: str = 'grid'):
        self.model_class = model_class
        self.search_space = search_space
        self.search_method = search_method
        self.best_params = None
        self.best_score = -float('inf')
        self.search_results = []
        # needed for dynamic data regeneration
        self.panel_data = None
        self.seq_len = None
        self.feature_cols = None
        self.split_ratios = (0.6, 0.2, 0.2)

    def generate_param_combinations(self, max_trials: int = 50):
        if self.search_method == 'grid':
            keys = list(self.search_space.keys())
            values = list(self.search_space.values())
            combos = list(product(*values))
            if len(combos) > max_trials:
                combos = combos[:max_trials]
            return [dict(zip(keys, c)) for c in combos]
        elif self.search_method == 'random':
            combos = []
            for _ in range(min(max_trials, 100)):
                combo = {k: np.random.choice(v) for k, v in self.search_space.items()}
                combos.append(combo)
            return combos
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")

    def _maybe_rebuild_datasets(self, params: Dict,
                                base_train: List[Data], base_val: List[Data]) -> Tuple[List[Data], List[Data]]:
        use_rebuild = (('corr_window' in params) or ('corr_threshold' in params)) and (self.panel_data is not None)
        if not use_rebuild:
            return base_train, base_val

        cw = max(2, int(params.get('corr_window', 5)))
        ct = float(params.get('corr_threshold', 0.3))

        ds, _ = create_enhanced_dataset(self.panel_data,
                                        seq_len=self.seq_len,
                                        corr_window=cw,
                                        corr_threshold=ct,
                                        feature_cols=self.feature_cols)

        if len(ds) < 10:
            return [], []

        n = len(ds)
        r1, r2, _ = self.split_ratios
        t_end = int(r1 * n)
        v_end = int((r1 + r2) * n)
        return ds[:t_end], ds[t_end:v_end]

    def evaluate_params(self, params: Dict, train_dataset: List[Data],
                        val_dataset: List[Data], fixed_params: Dict = None,
                        fast_epochs: int = 5, batch_size: int = 8, lr: float = 0.001) -> float:
        try:
            tr_ds, va_ds = self._maybe_rebuild_datasets(params, train_dataset, val_dataset)
            if len(tr_ds) == 0 or len(va_ds) == 0:
                return -1.0

            node_feat_dim = tr_ds[0].x.shape[1]

            full_params = dict(fixed_params or {})
            full_params.update(params)
            full_params['node_feat_dim'] = node_feat_dim
            full_params.pop('corr_window', None)
            full_params.pop('corr_threshold', None)
            model = self.model_class(**full_params)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            optim = torch.optim.Adam(model.parameters(), lr=lr)
            model.train()
            for _ in range(fast_epochs):  # a quick training pass
                for i in range(0, len(tr_ds), batch_size):
                    batch = tr_ds[i:i + batch_size]
                    optim.zero_grad()
                    try:
                        outputs = model(batch)
                        preds = outputs['predictions']                  # [B, A]
                        risks = outputs.get('risks', None)
                        if risks is not None: risks = risks.detach().to(device)  # cut gradients through the risk head
                        tgts  = torch.stack([d.y for d in batch]).to(device)     # [B, A]
                        s = _compute_signal(preds, risks)
                        corr = _spearman_corr(s, tgts, dim=1)
                        loss = (-corr).mean()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                    except Exception:
                        continue

            # validation score: average daily Spearman-IC (matches the training objective)
            ic_list = []
            model.eval()
            with torch.no_grad():
                for i in range(0, len(va_ds), batch_size):
                    batch = va_ds[i:i + batch_size]
                    try:
                        outputs = model(batch)
                        preds = outputs['predictions']
                        risks = outputs.get('risks', None)
                        if risks is not None: risks = risks.detach().to(device)  # validation uses the same convention
                        tgts  = torch.stack([d.y for d in batch]).to(device)
                        s = _compute_signal(preds, risks)
                        corr = _spearman_corr(s, tgts, dim=1)
                        ic_list.extend(corr.detach().cpu().tolist())
                    except Exception:
                        continue

            return float(np.nanmean(ic_list)) if ic_list else 0.0
        except Exception as e:
            print(f"Parameter evaluation failed {params}: {e}")
            return -1.0

    def search(self, train_dataset: List[Data], val_dataset: List[Data],
               fixed_params: Dict = None, max_trials: int = 20,
               panel_data: pd.DataFrame = None, seq_len: int = None,
               feature_cols: List[str] = None,
               fast_epochs: int = 5, batch_size: int = 8, lr: float = 0.001) -> Dict:
        print(f"Starting the hyperparameter search ({self.search_method})...")
        print(f"Search space: {self.search_space}")

        self.panel_data = panel_data
        self.seq_len = seq_len
        self.feature_cols = feature_cols

        combos = self.generate_param_combinations(max_trials)
        print(f"{len(combos)} parameter combinations total")

        for i, params in enumerate(combos):
            print(f"Evaluating combination {i + 1}/{len(combos)}: {params}")
            score = self.evaluate_params(params, train_dataset, val_dataset, fixed_params,
                                          fast_epochs=fast_epochs, batch_size=batch_size, lr=lr)
            self.search_results.append({'params': params, 'score': score, 'rank': i + 1})
            if score > self.best_score:
                self.best_score = score
                self.best_params = dict(params)
                print(f"Found a better parameter set, score={score:.4f}")

        self.search_results.sort(key=lambda x: x['score'], reverse=True)
        print("Hyperparameter search complete!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.search_results
        }
