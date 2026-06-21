"""
DamageArbiter — disagreement-driven arbitration.

Given the out-of-fold (OOF) predictions of the ViT-B/32 image-only baseline and
the CLIP-LLM multimodal baseline, this script:

  1. uses the shared prediction when ViT and CLIP agree;
  2. when they disagree, trains a lightweight logistic-regression arbitrator to
     decide which model to trust.

IMPORTANT — no ground-truth leakage. Every final arbitration feature is computed
from the base models' softmax outputs and is therefore available at inference on
new, unlabeled imagery:

    * confidence : max-softmax probability (probability of the predicted class)
    * entropy    : Shannon entropy of the softmax distribution
    * margin     : decision margin = top-1 minus top-2 probability

The quantity p_pred - p_true (saved as `error_margin` by the base-model scripts)
is NOT used here: it depends on the ground-truth label and is only a diagnostic
for the overconfident/ambiguous error analysis. The arbitrator's coefficients
and the decision threshold tau = 0.5 are fixed; at deployment the identical
feature computation is applied to new images, so no label is ever required.

Usage:
    python damage_arbiter.py \
        --vit-oof   oof_vitb32_all.csv \
        --clip-oof  oof_clip_gpt_all.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             matthews_corrcoef, f1_score as _f1)

N_SPLITS = 5
N_SEEDS = 10            # average over seeds for a stable estimate
TAU = 0.5              # fixed decision threshold (no per-config tuning)


def reconstruct_margin(p1, h):
    """Recover the 3-class decision margin (top1 - top2) from the max
    probability p1 and the entropy h, for base models whose OOF file stores
    only p_pred and entropy rather than the full softmax vector."""
    r = 1.0 - p1
    if r <= 1e-9:
        return p1
    h_rem = h + p1 * np.log(p1)           # = -p2 ln p2 - p3 ln p3
    def g(q):
        o = r - q
        a = -q * np.log(q) if q > 1e-12 else 0.0
        b = -o * np.log(o) if o > 1e-12 else 0.0
        return a + b
    lo, hi = r / 2, r
    if h_rem >= g(lo):
        q = r / 2
    elif h_rem <= g(hi):
        q = r
    else:
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if g(mid) > h_rem else (lo, mid)
        q = (lo + hi) / 2
    return max(p1 - q, 0.0)


def load_base(path):
    df = pd.read_csv(path)
    # entropy in the base-model scripts uses base-2; convert to nats so the
    # margin reconstruction (which assumes natural log) is consistent.
    if "entropy" in df.columns:
        df["entropy_nats"] = df["entropy"] * np.log(2)
    if "is_correct" not in df.columns:
        df["is_correct"] = (df["true"] == df["pred"]).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit-oof", required=True)
    ap.add_argument("--clip-oof", required=True)
    ap.add_argument("--out", default="damagearbiter_predictions.csv")
    args = ap.parse_args()

    vit = load_base(args.vit_oof)
    clip = load_base(args.clip_oof)
    vit["margin"] = [reconstruct_margin(p, h) for p, h in zip(vit["p_pred"], vit["entropy_nats"])]

    m = vit.merge(clip, on="path", suffixes=("_vit", "_clip"))
    assert (m["true_vit"] == m["true_clip"]).all(), "true labels differ between OOF files"
    true = m["true_vit"].values
    N = len(m)

    agree = m["pred_vit"].values == m["pred_clip"].values
    one_correct = (m["is_correct_vit"].values + m["is_correct_clip"].values) == 1
    arb_mask = (~agree) & one_correct          # disagreement, exactly one model right
    arb = m[arb_mask].copy()
    y = arb["is_correct_vit"].values           # 1 -> trust ViT, 0 -> trust CLIP

    print(f"Total {N} | agree {agree.sum()} | disagree {int((~agree).sum())} "
          f"| arbitrator-training {len(arb)} (ViT {int(y.sum())} / CLIP {int((1 - y).sum())})")

    def final_predictions(trust_vit):
        pred = m["pred_vit"].values.copy()
        for pos, t in zip(np.where(arb_mask)[0], trust_vit):
            pred[pos] = m["pred_vit"].values[pos] if t == 1 else m["pred_clip"].values[pos]
        return pred

    # ---- label-free feature blocks ----
    conf = np.column_stack([arb["confidence_vit"], arb["confidence_clip"]])      # max-softmax
    clip_probs = arb[["probs_0", "probs_1", "probs_2"]].values
    sp = np.sort(clip_probs, axis=1)
    clip_margin = sp[:, -1] - sp[:, -2]
    unc = np.column_stack([arb["entropy_vit"], arb["margin"],
                           arb["entropy_clip"], clip_margin])
    feats = {
        "LOGREG_conf": conf,
        "LOGREG_conf+unc": np.column_stack([conf, unc]),
    }

    # ---- ablation: each configuration, OOF, fixed tau, averaged over seeds ----
    print(f"\nAblation (fixed tau={TAU}, {N_SEEDS}-seed mean):")
    print(f"{'Configuration':<20}{'ArbAcc':>8}{'OverallAcc':>12}{'MCC':>8}")
    for name, X in feats.items():
        aa, oa, mc = [], [], []
        for s in range(N_SEEDS):
            skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=s)
            proba = cross_val_predict(
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                X, y, cv=skf, method="predict_proba")[:, 1]
            trust = (proba > TAU).astype(int)
            fp = final_predictions(trust)
            aa.append((trust == y).mean())
            oa.append((fp == true).mean())
            mc.append(matthews_corrcoef(true, fp))
        print(f"{name:<20}{np.mean(aa):>8.3f}{np.mean(oa):>12.4f}{np.mean(mc):>8.4f}")

    naive_trust = (arb["confidence_vit"].values > arb["confidence_clip"].values).astype(int)
    naive_pred = final_predictions(naive_trust)
    print(f"{'Naive heuristic':<20}{(naive_trust == y).mean():>8.3f}"
          f"{(naive_pred == true).mean():>12.4f}{matthews_corrcoef(true, naive_pred):>8.4f}")

    # ---- final arbitrator: confidence only (parsimonious, fully deployable) ----
    # The confidence configuration is selected for parsimony and deployability: all
    # learned configurations perform comparably (see ablation above), so the simplest
    # label-free feature set is preferred. Metrics are averaged over seeds for stability;
    # the saved prediction CSV uses the first seed.
    X = feats["LOGREG_conf"]
    acc, rec, pre, swf, mcc = [], [], [], [], []
    pred0 = None
    for s in range(N_SEEDS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=s)
        proba = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            X, y, cv=skf, method="predict_proba")[:, 1]
        pred = final_predictions((proba > TAU).astype(int))
        if s == 0:
            pred0 = pred
        acc.append((pred == true).mean())
        rec.append(recall_score(true, pred, average="weighted", zero_division=0))
        pre.append(precision_score(true, pred, average="weighted", zero_division=0))
        swf.append(f1_score(true, pred, average="weighted", zero_division=0))
        mcc.append(matthews_corrcoef(true, pred))
    print(f"\nFinal DamageArbiter (Confidence configuration, {N_SEEDS}-seed mean):")
    print(f"  Accuracy  {np.mean(acc):.4f}")
    print(f"  Recall    {np.mean(rec):.4f}")
    print(f"  Precision {np.mean(pre):.4f}")
    print(f"  SW-F1     {np.mean(swf):.4f}")
    print(f"  MCC       {np.mean(mcc):.4f}")

    out = m[["path"]].copy()
    out["true"] = true
    out["pred_vit"] = m["pred_vit"].values
    out["pred_clip"] = m["pred_clip"].values
    out["damagearbiter_pred"] = pred0
    out.to_csv(args.out, index=False)
    print(f"\nSaved final predictions -> {args.out}")


if __name__ == "__main__":
    main()
