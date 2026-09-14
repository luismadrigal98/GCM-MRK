"""Targeted diagnostics behind the manuscript's model-selection statements.

The benchmark drivers produce the paper's tables.  Several statements in the
text rest instead on smaller targeted experiments: that the optimizer returns
partitions scoring *above* the ground truth, that BIC recovers the factor rank,
that automatic rank selection loses nothing against an oracle rank, that graph
modularity is itself misspecified on the same processes, and the performance of
the two-stage recipe for choosing K.  Each is reproduced here so that every
statement in the text has a script behind it.

Experiments (each at the noise level the text reports for it)
------------------------------------------------------------------
1. Likelihood versus ground truth (sigma = 1.0), multi-factor and hub processes.
2. Rank recovery by BIC and AIC evaluated at the planted partition (sigma = 1.0).
3. Automatic versus oracle rank: for each process the factor model is fitted at
   q = 1, 2, 3; automatic selection keeps the lowest-BIC fit, the oracle keeps the
   highest-ARI one.  Both are read off the *same* fits (sigma = 1.0).
4. Graph modularity at the truth versus the partitions GCM and Leiden return
   (sigma = 1.0).  Newman modularity on the soft-thresholded |R|^6 graph -- note
   this is not the RBConfiguration objective Leiden itself optimizes.
5. Two-stage K selection on the five-module design (sigma = 1.5).

Outputs (../results):
  results/diagnostics.json          everything, per seed
  results/diagnostics_macros.tex    scalar macros for the manuscript

Run from repo root:
    PYTHONPATH=. python3 paper/experiments/diagnostics.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI

from gcmrk import cluster, cluster_factor_auto, simulate_modular_data
from gcmrk import metrics as M
from gcmrk.data import normalize_data, pearson_correlation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402
import misspecification as MS  # noqa: E402  (shares its generators and GA settings)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

GA = MS.GA                      # identical to the misspecification sweep
MODULES = MS.MODULES
N_SAMPLES = MS.N_SAMPLES
RANKS = (1, 2, 3)
TRUE_RANK = {"onefactor": 1, "multifactor": 3, "hub": 1}


def data(gen, noise, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, t = MS.GENERATORS[gen](MODULES, noise, seed)
    cor = pearson_correlation(normalize_data(X, by_sample=True), rowvar=True)
    return X, t, cor


def newman_modularity(cor, labels, power=B.SOFT_POWER):
    A = B.adjacency(cor, power)
    k = A.sum(1)
    m2 = A.sum()
    labels = np.asarray(labels)
    q = 0.0
    for c in np.unique(labels):
        f = labels == c
        q += A[np.ix_(f, f)].sum() / m2 - (k[f].sum() / m2) ** 2
    return float(q)


# --------------------------------------------------------------------------- #
def exp_likelihood(n_seeds):
    """Does the optimizer's partition outscore the planted truth?"""
    out = {}
    for gen in ("multifactor", "hub"):
        rows = []
        for s in range(n_seeds):
            X, t, cor = data(gen, 1.0, s)
            lab = cluster(X, targets=["loglik"], g_max=len(np.unique(t)), **GA).labels
            rows.append({"seed": s,
                         "ll_gcm": float(M.log_likelihood_correlation(cor, lab)),
                         "ll_truth": float(M.log_likelihood_correlation(cor, t)),
                         "ari": float(ARI(t, lab))})
        out[gen] = rows
        wins = sum(r["ll_gcm"] > r["ll_truth"] for r in rows)
        print(f"  [1] {gen}: GCM outscores truth in {wins}/{n_seeds}; "
              f"mean ARI {np.mean([r['ari'] for r in rows]):.3f}", flush=True)
    return out


def exp_rank_at_truth(n_seeds):
    """Which rank does each criterion pick when evaluated at the planted partition?"""
    out = {}
    for gen, q_true in TRUE_RANK.items():
        picks = {"bic": [], "aic": []}
        for s in range(n_seeds):
            X, t, cor = data(gen, 1.0, s)
            d = X.shape[1]
            for crit, fn in (("bic", M.factor_bic), ("aic", M.factor_aic)):
                scores = {q: fn(cor, t, d, q=q) for q in (1, 2, 3, 4)}
                picks[crit].append(int(min(scores, key=scores.get)))
        out[gen] = {"true_rank": q_true, **picks}
        print(f"  [2] {gen}: true q={q_true}  BIC picks {picks['bic']}  "
              f"AIC picks {picks['aic']}", flush=True)
    return out


def exp_auto_vs_oracle(n_seeds):
    """Automatic (lowest-BIC) rank versus the best rank chosen by ARI, same fits."""
    out = {}
    for gen in ("onefactor", "multifactor", "hub", "network"):
        rows = []
        for s in range(n_seeds):
            X, t, cor = data(gen, 1.0, s)
            k = len(np.unique(t))
            fits = {}
            for q in RANKS:
                r = cluster(X, targets=[f"loglik_factor_bic@{q}"], g_max=k, **GA)
                fits[q] = {"bic": float(r.scores[f"loglik_factor_bic@{q}"]),
                           "ari": float(ARI(t, r.labels))}
            q_auto = min(fits, key=lambda q: fits[q]["bic"])
            q_best = max(fits, key=lambda q: fits[q]["ari"])
            loglik_ari = float(ARI(t, cluster(X, targets=["loglik"], g_max=k, **GA).labels))
            rows.append({"seed": s, "fits": {str(q): v for q, v in fits.items()},
                         "q_auto": q_auto, "q_best": q_best,
                         "ari_auto": fits[q_auto]["ari"],
                         "ari_best": fits[q_best]["ari"],
                         "ari_loglik": loglik_ari})
        out[gen] = rows
        gap = np.mean([r["ari_best"] - r["ari_auto"] for r in rows])
        print(f"  [3] {gen}: auto ARI {np.mean([r['ari_auto'] for r in rows]):.3f}  "
              f"oracle-rank ARI {np.mean([r['ari_best'] for r in rows]):.3f}  "
              f"(gap {gap:.3f})  q=3 ARI "
              f"{np.mean([r['fits']['3']['ari'] for r in rows]):.3f}", flush=True)
    return out


def exp_modularity(n_seeds):
    """Does Newman modularity rank the planted truth above GCM's and Leiden's partitions?"""
    out = {}
    for gen in ("onefactor", "multifactor", "hub"):
        rows = []
        for s in range(n_seeds):
            X, t, cor = data(gen, 1.0, s)
            k = len(np.unique(t))
            Xn = normalize_data(X, by_sample=True)
            gcm = cluster(X, targets=["loglik"], g_max=k, **GA).labels
            lei = B.leiden(cor, Xn, k, np.random.default_rng(0))
            q_t, q_g, q_l = (newman_modularity(cor, lab) for lab in (t, gcm, lei))
            rows.append({"seed": s, "q_truth": q_t, "q_gcm": q_g, "q_leiden": q_l,
                         "truth_top": bool(q_t >= max(q_g, q_l) - 1e-12)})
        out[gen] = rows
        print(f"  [4] {gen}: modularity ranks truth top in "
              f"{sum(r['truth_top'] for r in rows)}/{n_seeds}", flush=True)
    return out


def exp_two_stage(n_seeds, g_loose=10):
    """loglik_aic chooses K; the factor model is then fitted with K held there."""
    rows = []
    k_true = len(MODULES)
    for s in range(n_seeds):
        X, t = simulate_modular_data(MODULES, n_samples=N_SAMPLES, noise=1.5,
                                     latent_per_module=1, seed=s)
        r1 = cluster(X, targets=["loglik_aic"], g_max=g_loose, **GA)
        k_hat = len(np.unique(r1.labels))
        r2 = cluster_factor_auto(X, ranks=RANKS, g_max=k_hat, **GA)
        r3 = cluster_factor_auto(X, ranks=RANKS, g_max=g_loose, **GA)
        r4 = cluster_factor_auto(X, ranks=RANKS, g_max=k_true, **GA)
        rows.append({"seed": s, "k_hat": k_hat,
                     "ari_aic": float(ARI(t, r1.labels)),
                     "ari_two_stage": float(ARI(t, r2.labels)),
                     "ari_loose": float(ARI(t, r3.labels)),
                     "k_loose": int(len(np.unique(r3.labels))),
                     "ari_oracle_k": float(ARI(t, r4.labels))})
    for key in ("ari_aic", "ari_two_stage", "ari_loose", "ari_oracle_k"):
        print(f"  [5] {key:14s} {np.mean([r[key] for r in rows]):.3f}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
def write_macros(d):
    def mac(n, v):
        return f"\\newcommand{{\\{n}}}{{{v}}}\n"
    m = ""

    ll = d["likelihood"]
    m += mac("DiagLLSeeds", len(ll["multifactor"]))
    for gen, short in (("multifactor", "Multi"), ("hub", "Hub")):
        rows = ll[gen]
        m += mac(f"DiagLL{short}Wins",
                 f"{sum(r['ll_gcm'] > r['ll_truth'] for r in rows)}/{len(rows)}")
        m += mac(f"DiagLL{short}ARI", f"{np.mean([r['ari'] for r in rows]):.2f}")
    ex = ll["multifactor"][0]
    m += mac("DiagLLExampleGCM", f"{ex['ll_gcm']:.1f}")
    m += mac("DiagLLExampleTruth", f"{ex['ll_truth']:.1f}")

    rk = d["rank_at_truth"]
    m += mac("DiagRankSeeds", len(rk["onefactor"]["bic"]))
    for gen, short in (("onefactor", "OneFac"), ("multifactor", "MultiFac"),
                       ("hub", "Hub")):
        q = rk[gen]["true_rank"]
        for crit, cshort in (("bic", "Bic"), ("aic", "Aic")):
            picks = rk[gen][crit]
            m += mac(f"DiagRank{cshort}{short}",
                     f"{sum(p == q for p in picks)}/{len(picks)}")

    av = d["auto_vs_oracle"]
    all_rows = [r for rows in av.values() for r in rows]
    m += mac("DiagAutoSeeds", len(av["onefactor"]))
    m += mac("DiagAutoMatches",
             f"{sum(abs(r['ari_best'] - r['ari_auto']) < 1e-9 for r in all_rows)}"
             f"/{len(all_rows)}")
    m += mac("DiagAutoGap",
             f"{np.mean([r['ari_best'] - r['ari_auto'] for r in all_rows]):.3f}")
    one = av["onefactor"]
    m += mac("DiagOneFacLoglikARI", f"{np.mean([r['ari_loglik'] for r in one]):.2f}")
    m += mac("DiagOneFacQthreeARI",
             f"{np.mean([r['fits']['3']['ari'] for r in one]):.2f}")

    mo = d["modularity"]
    m += mac("DiagModSeeds", len(mo["onefactor"]))
    for gen, short in (("onefactor", "OneFac"), ("multifactor", "MultiFac"),
                       ("hub", "Hub")):
        rows = mo[gen]
        m += mac(f"DiagModTruthTop{short}",
                 f"{sum(r['truth_top'] for r in rows)}/{len(rows)}")

    ts = d["two_stage"]
    m += mac("TwoStageSeeds", len(ts))
    m += mac("TwoStageK", f"{np.mean([r['k_hat'] for r in ts]):.1f}")
    m += mac("TwoStageARI", f"{np.mean([r['ari_two_stage'] for r in ts]):.2f}")
    m += mac("TwoStageAicARI", f"{np.mean([r['ari_aic'] for r in ts]):.2f}")
    m += mac("TwoStageLooseARI", f"{np.mean([r['ari_loose'] for r in ts]):.2f}")
    m += mac("TwoStageLooseK", f"{np.mean([r['k_loose'] for r in ts]):.1f}")
    m += mac("TwoStageOracleARI", f"{np.mean([r['ari_oracle_k'] for r in ts]):.2f}")
    (RESULTS / "diagnostics_macros.tex").write_text(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 seeds everywhere")
    args = ap.parse_args()
    n10, n6, n5 = (2, 2, 2) if args.quick else (10, 6, 5)

    t0 = time.time()
    d = {"meta": {"ga": GA, "modules": MODULES, "n_samples": N_SAMPLES,
                  "ranks": list(RANKS), "quick": args.quick}}
    print("[1] likelihood versus ground truth", flush=True)
    d["likelihood"] = exp_likelihood(n10)
    print("[2] rank recovery at the planted partition", flush=True)
    d["rank_at_truth"] = exp_rank_at_truth(n10)
    print("[3] automatic versus oracle rank", flush=True)
    d["auto_vs_oracle"] = exp_auto_vs_oracle(n5)
    print("[4] modularity at the truth", flush=True)
    d["modularity"] = exp_modularity(n6)
    print("[5] two-stage K selection", flush=True)
    d["two_stage"] = exp_two_stage(n10)
    d["meta"]["seconds"] = round(time.time() - t0, 1)

    (RESULTS / "diagnostics.json").write_text(json.dumps(d, indent=1))
    if not args.quick:
        write_macros(d)
    print(f"\nDone in {time.time()-t0:.0f}s -> {RESULTS / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
