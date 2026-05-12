"""Plot image-level damage severity by disaster semantic probe category.

The script expects two CSV files with aligned rows from CLIP-Human and
CLIP-LLM semantic scoring. Required columns are:

    path, lat, lon, true, sim_flood, sim_tree, sim_debris, sim_infra

Point color encodes the image-level ground-truth damage severity
(mild / moderate / severe). Each panel highlights samples with relatively
strong semantic salience for one probe category, using the average of the
CLIP-Human and CLIP-LLM similarity scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


PROBES = [
    ("Flood", "sim_flood"),
    ("Tree", "sim_tree"),
    ("Debris", "sim_debris"),
    ("Infrastructure", "sim_infra"),
]

SEVERITY_ORDER = ["mild", "moderate", "severe"]
SEVERITY_COLORS = {
    "mild": "#66CC33",
    "moderate": "#F2B200",
    "severe": "#D7191C",
}


def minmax(values: pd.Series) -> pd.Series:
    low = float(values.min())
    high = float(values.max())
    if np.isclose(low, high):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def robust_salience(values: pd.Series, lower: float = 0.03, upper: float = 0.97) -> pd.Series:
    """Normalize semantic scores while reducing the influence of outliers."""
    low = float(values.quantile(lower))
    high = float(values.quantile(upper))
    return minmax(values.clip(low, high))


def load_probe_data(human_csv: Path, llm_csv: Path) -> pd.DataFrame:
    human = pd.read_csv(human_csv)
    llm = pd.read_csv(llm_csv)

    if not human["path"].equals(llm["path"]):
        raise ValueError("Human and LLM CSV rows are not aligned by path.")
    if not human[["lat", "lon"]].equals(llm[["lat", "lon"]]):
        raise ValueError("Human and LLM CSV rows are not aligned by coordinates.")

    required = {"lat", "lon", "true", *(column for _, column in PROBES)}
    missing = sorted(required - set(human.columns))
    if missing:
        raise ValueError(f"Missing required columns in human CSV: {missing}")
    missing = sorted(required - set(llm.columns))
    if missing:
        raise ValueError(f"Missing required columns in LLM CSV: {missing}")

    out = human[["path", "lat", "lon", "true"]].copy()
    out = out.rename(columns={"true": "severity"})
    unknown = sorted(set(out["severity"].dropna()) - set(SEVERITY_ORDER))
    if unknown:
        raise ValueError(f"Unexpected severity labels: {unknown}")

    for label, column in PROBES:
        semantic_mean = (human[column] + llm[column]) / 2.0
        out[f"{label} Salience"] = robust_salience(semantic_mean)
        out[f"{label} Semantic Mean"] = semantic_mean

    return out


def make_local_segments(df: pd.DataFrame, max_distance: float = 0.00012) -> list[np.ndarray]:
    """Build a subtle street skeleton from very local neighboring samples."""
    coords = df[["lon", "lat"]].to_numpy()
    pairs: set[tuple[int, int]] = set()
    for i, point in enumerate(coords):
        deltas = coords - point
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        nearest = np.argsort(distances)[1:3]
        for j in nearest:
            if distances[j] <= max_distance:
                pairs.add(tuple(sorted((i, int(j)))))
    return [coords[list(pair)] for pair in sorted(pairs)]


def plot_probe_maps(df: pd.DataFrame, output: Path, salience_quantile: float = 0.45) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
        }
    )

    lon_pad = (df["lon"].max() - df["lon"].min()) * 0.035
    lat_pad = (df["lat"].max() - df["lat"].min()) * 0.055
    lon_limits = (df["lon"].min() - lon_pad, df["lon"].max() + lon_pad)
    lat_limits = (df["lat"].min() - lat_pad, df["lat"].max() + lat_pad)
    mean_lat = np.deg2rad(df["lat"].mean())
    segments = make_local_segments(df)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.45), sharex=True, sharey=True)
    axes = axes.ravel()
    panel_letters = ["a", "b", "c", "d"]

    for idx, (ax, letter, (label, _)) in enumerate(zip(axes, panel_letters, PROBES)):
        ax.set_facecolor("#fbfbf8")
        ax.add_collection(LineCollection(segments, colors="#cfcfcf", linewidths=0.7, alpha=0.72, zorder=1))
        ax.scatter(df["lon"], df["lat"], s=5.0, color="#d0d0d0", alpha=0.32, linewidths=0, zorder=2)

        salience = df[f"{label} Salience"]
        highlighted = df.assign(_salience=salience)
        highlighted = highlighted.loc[salience >= salience.quantile(salience_quantile)].sort_values("_salience")
        colors = [SEVERITY_COLORS[severity] for severity in highlighted["severity"]]

        ax.scatter(
            highlighted["lon"],
            highlighted["lat"],
            c=colors,
            s=8.2,
            edgecolors="#1a1a1a",
            linewidths=0.22,
            alpha=0.96,
            zorder=3,
        )

        ax.set_xlim(lon_limits)
        ax.set_ylim(lat_limits)
        ax.set_aspect(1 / np.cos(mean_lat))
        ax.grid(True, color="#e1e1dc", linewidth=0.45, alpha=0.7)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3f"))

        if idx >= 2:
            ax.set_xlabel("Longitude", labelpad=4)
        if idx % 2 == 0:
            ax.set_ylabel("Latitude", labelpad=4)
        else:
            ax.tick_params(labelleft=False)

        ax.set_title(f"({letter}) {label}", loc="left", fontsize=8.5, fontweight="normal", pad=4)
        for spine in ax.spines.values():
            spine.set_color("#2f2f2f")
            spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.08, right=0.82, top=0.955, bottom=0.105, wspace=0.16, hspace=0.2)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SEVERITY_COLORS[label],
            markeredgecolor="none",
            markersize=6.2,
            label=label.capitalize(),
        )
        for label in SEVERITY_ORDER
    ]
    fig.legend(
        handles=handles,
        title="Damage severity",
        loc="center left",
        bbox_to_anchor=(0.845, 0.53),
        frameon=False,
        fontsize=7.1,
        title_fontsize=7.4,
        labelspacing=0.7,
        handletextpad=0.6,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-csv", required=True, type=Path, help="CSV with CLIP-Human semantic probe scores.")
    parser.add_argument("--llm-csv", required=True, type=Path, help="CSV with CLIP-LLM semantic probe scores.")
    parser.add_argument(
        "--output",
        default=Path("figure/figure11.semantic_probe_severity_maps.png"),
        type=Path,
        help="Output PNG path. A PDF with the same stem is also written.",
    )
    parser.add_argument(
        "--salience-quantile",
        default=0.45,
        type=float,
        help="Lower quantile cutoff for highlighting probe-salient samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_probe_data(args.human_csv, args.llm_csv)
    plot_probe_maps(df, args.output, salience_quantile=args.salience_quantile)
    print(f"Wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
