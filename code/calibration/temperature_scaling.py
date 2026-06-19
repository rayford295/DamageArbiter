"""
Post-hoc confidence calibration via temperature scaling (Section 4.3.4).

Temperature scaling rescales a model's logits by a single scalar T before the
softmax:  p_i = softmax(z_i / T).  Because dividing all logits by the same T
preserves their ranking, the predicted class (and therefore the accuracy) is
unchanged; only the confidence values are recalibrated.  T is fit on a held-out
(out-of-fold) validation set by minimizing the negative log-likelihood.

Input:  an out-of-fold predictions CSV with columns
    true        ground-truth label (mild/moderate/severe or 0/1/2)
    probs_0, probs_1, probs_2   predicted class probabilities
    fold        (optional) cross-validation fold id for out-of-fold T fitting

Reported: fitted temperature T, expected calibration error (ECE) before/after,
and the proportion of overconfident errors before/after, where an error is
"overconfident" when (p_predicted_class - p_true_class) > 0.4.

Usage:
    python temperature_scaling.py --oof oof_clip_gpt_all.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

LABELS = {"mild": 0, "moderate": 1, "severe": 2}


def _encode(col):
    mapped = col.astype(str).str.strip().map(LABELS)
    if mapped.notna().all():
        return mapped.astype(int).values
    return pd.to_numeric(col).astype(int).values


def scale(probs, T):
    """Apply temperature T to a probability matrix (via implied logits)."""
    z = np.log(np.clip(probs, 1e-12, 1.0))
    zt = z / T
    zt -= zt.max(axis=1, keepdims=True)
    e = np.exp(zt)
    return e / e.sum(axis=1, keepdims=True)


def nll(T, probs, y):
    p = scale(probs, T)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


def ece(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(1.0 / probs.shape[1], 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.sum() / len(y) * abs(acc[m].mean() - conf[m].mean())
    return e


def overconfident_rate(probs, y, thr=0.4):
    pred = probs.argmax(1)
    wrong = pred != y
    if wrong.sum() == 0:
        return 0.0
    margin = probs[wrong, pred[wrong]] - probs[wrong, y[wrong]]
    return 100.0 * (margin > thr).mean()


def fit_temperature(probs, y, folds=None):
    """Fit T. If fold ids are given, fit per-fold out-of-fold and average."""
    if folds is None:
        return minimize_scalar(lambda T: nll(T, probs, y),
                               bounds=(0.5, 5.0), method="bounded").x
    T_oof = np.zeros(len(y))
    for f in np.unique(folds):
        tr = folds != f
        Tf = minimize_scalar(lambda T: nll(T, probs[tr], y[tr]),
                             bounds=(0.5, 5.0), method="bounded").x
        T_oof[folds == f] = Tf
    return T_oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="out-of-fold predictions CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.oof)
    y = _encode(df["true"])
    probs = df[["probs_0", "probs_1", "probs_2"]].values
    folds = df["fold"].values if "fold" in df.columns else None

    T = fit_temperature(probs, y, folds)
    cal = scale(probs, T[:, None] if np.ndim(T) else T)

    T_report = float(np.mean(T)) if np.ndim(T) else float(T)
    print(f"Fitted temperature T = {T_report:.3f} (T > 1 indicates overconfidence)")
    print(f"ECE                : {ece(probs, y):.4f} -> {ece(cal, y):.4f}")
    print(f"Overconfident errors: {overconfident_rate(probs, y):.2f}% "
          f"-> {overconfident_rate(cal, y):.2f}%")
    # Accuracy is unchanged because temperature scaling preserves the argmax.
    acc0 = (probs.argmax(1) == y).mean()
    acc1 = (cal.argmax(1) == y).mean()
    print(f"Accuracy (unchanged): {acc0:.4f} -> {acc1:.4f}")


if __name__ == "__main__":
    main()
