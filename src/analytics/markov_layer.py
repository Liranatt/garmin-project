"""Markov and KL-divergence layer helpers for correlation analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


def compute_markov_layer(
    *,
    data: pd.DataFrame,
    metrics: Sequence[str],
    normality: Dict[str, Any],
    key_targets: Sequence[str],
    rename_map: Dict[str, str],
    n_bins: int,
    bin_labels: Sequence[str],
    acwr_edges: Sequence[float],
    acwr_labels: Sequence[str],
    min_transitions_kl: int,
    adaptive_alpha_fn,
    logger,
) -> List[Dict[str, Any]]:
    """Compute Markov transition matrices with conditional KL ranking."""
    if "date" not in data.columns:
        raise ValueError("Markov layer requires a 'date' column in input data.")

    logger.info("   Layer 2e: Markov transitions + KL...")

    markov_targets = [
        m
        for m in metrics
        if normality.get(m, ("normal",))[0] == "non_normal" and m in data.columns and data[m].notna().sum() >= 8
    ]
    for m in key_targets:
        actual = rename_map.get(m, m)
        if actual in metrics and actual not in markov_targets and actual in data.columns and data[actual].notna().sum() >= 8:
            markov_targets.append(actual)

    results: List[Dict[str, Any]] = []
    for target in markov_targets:
        if target not in data.columns:
            continue

        non_null = data[["date", target]].dropna()
        vals = non_null[target].values
        dates = pd.to_datetime(non_null["date"]).values
        if len(vals) < 6:
            continue

        if target == "acwr":
            edges = np.array(acwr_edges, dtype=np.float64)
            actual_labels = list(acwr_labels)
        else:
            edges = np.percentile(vals, np.linspace(0, 100, n_bins + 1))
            edges[0] -= 1
            edges[-1] += 1
            actual_labels = list(bin_labels)
        edges = np.unique(edges)
        actual_bins = len(edges) - 1
        if actual_bins < 2:
            continue

        bins = np.clip(np.digitize(vals, edges[1:-1]), 0, actual_bins - 1)

        t_marginal = np.zeros((actual_bins, actual_bins), dtype=np.float64)
        for t in range(len(bins) - 1):
            day_gap = (dates[t + 1] - dates[t]) / np.timedelta64(1, "D")
            if day_gap == 1:
                t_marginal[bins[t], bins[t + 1]] += 1

        n_trans = int(t_marginal.sum())
        alpha = adaptive_alpha_fn(n_trans)

        rs = t_marginal.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        t_marginal_norm = t_marginal / rs
        uniform = np.ones_like(t_marginal_norm) / actual_bins
        t_marginal_norm = (1 - alpha) * t_marginal_norm + alpha * uniform
        rs = t_marginal_norm.sum(axis=1, keepdims=True)
        t_marginal_norm = t_marginal_norm / rs

        try:
            eigvals, eigvecs = np.linalg.eig(t_marginal_norm.T)
            idx = np.argmin(np.abs(eigvals - 1.0))
            stationary = np.real(eigvecs[:, idx])
            stationary = stationary / stationary.sum()
        except Exception:
            stationary = np.ones(actual_bins) / actual_bins

        best_kl = 0.0
        best_cond = None
        best_cond_t = None

        cond_cands = [
            m
            for m in metrics
            if m != target and m in data.columns and data[m].notna().sum() >= 8
        ]

        for cond_metric in cond_cands:
            sub = data[["date", target, cond_metric]].dropna()
            if len(sub) < 6:
                continue

            t_vals = sub[target].values
            c_vals = sub[cond_metric].values
            sub_dates = pd.to_datetime(sub["date"]).values

            t_bins = np.clip(np.digitize(t_vals, edges[1:-1]), 0, actual_bins - 1)
            c_med = np.median(c_vals)
            c_bins = (c_vals > c_med).astype(int)

            t_cond: Dict[int, np.ndarray] = {}
            level_trans: Dict[int, int] = {}

            for c_level in [0, 1]:
                t_c = np.zeros((actual_bins, actual_bins), dtype=np.float64)
                for t in range(len(t_bins) - 1):
                    day_gap = (sub_dates[t + 1] - sub_dates[t]) / np.timedelta64(1, "D")
                    if c_bins[t] == c_level and day_gap == 1:
                        t_c[t_bins[t], t_bins[t + 1]] += 1

                n_trans_c = int(t_c.sum())
                level_trans[c_level] = n_trans_c

                rs = t_c.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1
                t_c = t_c / rs
                alpha_c = adaptive_alpha_fn(n_trans_c)
                t_c = (1 - alpha_c) * t_c + alpha_c * uniform
                rs = t_c.sum(axis=1, keepdims=True)
                t_c = t_c / rs
                t_cond[c_level] = t_c

            if min(level_trans.values()) < min_transitions_kl:
                continue

            eps = 1e-12
            kl_total = 0.0
            for c_level in [0, 1]:
                p = np.clip(t_cond[c_level], eps, None)
                q = np.clip(t_marginal_norm, eps, None)
                kl = float(np.mean(np.sum(p * np.log(p / q), axis=1)))
                kl_total += kl
            kl_avg = kl_total / 2

            if kl_avg > best_kl:
                best_kl = kl_avg
                best_cond = cond_metric
                best_cond_t = t_cond

        n_total = int(t_marginal.sum())
        if n_total >= 100:
            conf_tier = "HIGH"
        elif n_total >= 30:
            conf_tier = "GOOD"
        elif n_total >= 15:
            conf_tier = "MODERATE"
        else:
            conf_tier = "PRELIMINARY"

        results.append(
            {
                "target": target,
                "bins": actual_bins,
                "labels": actual_labels[:actual_bins],
                "edges": edges,
                "marginal": t_marginal_norm,
                "stationary": stationary,
                "n_transitions": n_total,
                "smooth_alpha": alpha,
                "confidence": conf_tier,
                "best_cond": best_cond,
                "best_cond_T": best_cond_t,
                "best_kl": best_kl,
            }
        )

    results.sort(key=lambda x: x["best_kl"], reverse=True)
    logger.info("   OK %d Markov models", len(results))
    return results

