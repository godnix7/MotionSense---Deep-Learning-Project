"""Plotly figures (in-app) and Matplotlib exports. All charts read the saved experiment files
(training history, test metrics, class distribution) — nothing is synthesised here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from utils.config import DISTRIBUTION_PATH, HISTORY_PATH, TEST_METRICS_PATH

INK = "#202B38"
MUTED = "#687586"
GRID = "#E3E7ED"
BLUE = "#1769AA"
TEAL = "#16A6A1"
NAVY = "#14263D"
AMBER = "#E6A23C"
GREEN = "#239B75"
UNKNOWN_GREY = "#B4BCC7"
# distinct, restrained categorical palette for the six activities
CLASS_COLORS = ["#1769AA", "#16A6A1", "#E6A23C", "#7A5CC2", "#D1495B", "#4F6D7A"]
MONO = "IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
SANS = "Inter, Helvetica Neue, Arial, sans-serif"


def load_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def base_layout(fig: go.Figure, height: int = 320, title: str | None = None, legend: bool = True) -> go.Figure:
    # Titles are provided by the page's section headers; in-chart titles collided with legends.
    fig.update_layout(
        height=height, margin=dict(l=56, r=20, t=40 if legend else 12, b=48),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(family=SANS, size=12, color=INK),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=MONO, size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor="#C9D0D9", ticks="outside",
                     tickcolor="#C9D0D9", tickfont=dict(family=MONO, size=11, color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, linecolor="#C9D0D9", ticks="outside",
                     tickcolor="#C9D0D9", tickfont=dict(family=MONO, size=11, color=MUTED))
    return fig


def color_map(classes: list[str]) -> dict:
    m = {c: CLASS_COLORS[i % len(CLASS_COLORS)] for i, c in enumerate(classes)}
    m["UNKNOWN / UNCERTAIN ACTIVITY"] = UNKNOWN_GREY
    return m


# ---- training curves ------------------------------------------------------------------------
def loss_curve(history: list[dict]) -> go.Figure:
    ep = [h["epoch"] for h in history]
    fig = go.Figure()
    fig.add_scatter(x=ep, y=[h["train_loss"] for h in history], name="Training loss", mode="lines+markers",
                    line=dict(color=BLUE, width=2), marker=dict(size=5))
    fig.add_scatter(x=ep, y=[h["val_loss"] for h in history], name="Validation loss", mode="lines+markers",
                    line=dict(color=TEAL, width=2, dash="dash"), marker=dict(size=5))
    fig.update_xaxes(title_text="Epoch", dtick=2 if len(ep) <= 30 else None)
    fig.update_yaxes(title_text="Cross-entropy loss")
    return base_layout(fig, title="LOSS")


def accuracy_curve(history: list[dict]) -> go.Figure:
    ep = [h["epoch"] for h in history]
    fig = go.Figure()
    fig.add_scatter(x=ep, y=[h["train_acc"] for h in history], name="Training accuracy (clip)", mode="lines+markers",
                    line=dict(color=BLUE, width=2), marker=dict(size=5))
    fig.add_scatter(x=ep, y=[h["val_clip_acc"] for h in history], name="Validation accuracy (clip)",
                    mode="lines+markers", line=dict(color=TEAL, width=2, dash="dash"), marker=dict(size=5))
    fig.add_scatter(x=ep, y=[h["val_video_acc"] for h in history], name="Validation accuracy (video)",
                    mode="lines", line=dict(color=AMBER, width=1.5, dash="dot"))
    fig.update_xaxes(title_text="Epoch", dtick=2 if len(ep) <= 30 else None)
    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", range=[0, 1.02])
    return base_layout(fig, title="ACCURACY")


def confusion_matrix_fig(cm: list[list[int]], names: list[str], normalize: bool = True) -> go.Figure:
    cm = np.array(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    z = norm if normalize else cm
    text = [[f"{int(cm[i, j])}<br><span style='font-size:10px'>{norm[i, j]:.0%}</span>" for j in range(len(names))]
            for i in range(len(names))]
    # two-line tick labels keep the axis horizontal and readable (no 35° rotation)
    short = [n.replace(" (", "<br>(").replace("Bodyweight ", "Bodyweight<br>").replace("Jumping ", "Jumping<br>")
             for n in names]
    fig = go.Figure(go.Heatmap(
        z=z, x=short, y=names, text=text, texttemplate="%{text}", colorscale=[[0, "#FFFFFF"], [0.5, "#8FB8DA"], [1, NAVY]],
        zmin=0, zmax=1 if normalize else None, showscale=False, xgap=3, ygap=3,
        hovertemplate="True %{y}<br>Predicted %{x}<br>%{z:.2f}<extra></extra>",
        textfont=dict(family=MONO, size=12)))
    fig.update_yaxes(autorange="reversed", title_text="True class", showgrid=False, ticks="")
    fig.update_xaxes(title_text="Predicted class", showgrid=False, side="bottom", tickangle=0, ticks="")
    fig = base_layout(fig, height=500, title="CONFUSION MATRIX — OFFICIAL TEST SPLIT", legend=False)
    fig.update_yaxes(tickfont=dict(family=SANS, size=11.5, color=MUTED))   # after base_layout (it sets mono ticks)
    fig.update_xaxes(tickfont=dict(family=SANS, size=11, color=MUTED))
    fig.update_layout(margin=dict(l=130, r=16, t=16, b=78))
    # dark cells need light text
    fig.update_traces(textfont_color=None)
    return fig


def distribution_fig(dist: dict, names: dict) -> go.Figure:
    classes = list(dist["counts"].keys())
    labels = [names.get(c, c) for c in classes]
    fig = go.Figure()
    for subset, col in (("train", BLUE), ("val", TEAL), ("test", AMBER)):
        fig.add_bar(y=labels, x=[dist["counts"][c].get(subset, 0) for c in classes], name=subset.capitalize(),
                    orientation="h", marker_color=col, marker_line_width=0)
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(title_text="Videos")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return base_layout(fig, height=300, title="DATASET CLASS DISTRIBUTION (VIDEOS PER SUBSET)")


def per_class_fig(report: dict, classes: list[str], names: dict) -> go.Figure:
    labels = [names.get(c, c).replace(" (", "<br>(").replace("Bodyweight ", "Bodyweight<br>").replace("Jumping ", "Jumping<br>")
              for c in classes]
    fig = go.Figure()
    for metric, col in (("precision", BLUE), ("recall", TEAL), ("f1-score", AMBER)):
        fig.add_bar(x=labels, y=[report[c][metric] for c in classes], name=metric.capitalize(), marker_color=col)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(showgrid=False, tickangle=0)
    fig = base_layout(fig, height=340, title="PER-CLASS METRICS — OFFICIAL TEST SPLIT")
    fig.update_xaxes(tickfont=dict(family=SANS, size=11, color=MUTED))
    return fig


# ---- timelines ------------------------------------------------------------------------------
def segment_timeline(segments: list[dict], classes: list[str], names: dict, x_title="Time (s)",
                     height: int = 150, cursor: float | None = None) -> go.Figure:
    cmap = color_map(classes)
    fig = go.Figure()
    seen = set()
    for g in segments:
        lab = g["label"]
        nm = names.get(lab, lab) if lab in classes else "Uncertain"
        fig.add_bar(y=["Activity"], x=[max(g["end"] - g["start"], 0.05)], base=[g["start"]], orientation="h",
                    marker=dict(color=cmap.get(lab, UNKNOWN_GREY), line=dict(width=1, color="#FFFFFF")),
                    name=nm, legendgroup=nm, showlegend=nm not in seen, width=0.6,
                    hovertemplate=f"{nm}<br>%{{base:.1f}}s – {g['end']:.1f}s<br>confidence {g['confidence']:.0%}<extra></extra>")
        seen.add(nm)
    if cursor is not None:
        fig.add_vline(x=cursor, line=dict(color=INK, width=2))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text=x_title, rangemode="tozero")
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return base_layout(fig, height=height, legend=True)


def probability_timeline(times: list[float], probs: list[list[float]], classes: list[str], names: dict,
                         threshold: float | None = None, height: int = 260, cursor: float | None = None) -> go.Figure:
    cmap = color_map(classes)
    P = np.array(probs) if len(probs) else np.zeros((0, len(classes)))
    fig = go.Figure()
    for i, c in enumerate(classes):
        fig.add_scatter(x=times, y=P[:, i] if len(P) else [], name=names.get(c, c), mode="lines",
                        line=dict(color=cmap[c], width=2, shape="spline", smoothing=0.6))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=MUTED, width=1, dash="dot"),
                      annotation_text=f"threshold {threshold:.2f}", annotation_position="top left",
                      annotation_font=dict(family=MONO, size=10, color=MUTED))
    if cursor is not None:
        fig.add_vline(x=cursor, line=dict(color=INK, width=2))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", title_text="Class probability")
    fig.update_xaxes(title_text="Time (s)")
    return base_layout(fig, height=height)


def prob_bars(probs: list[float], classes: list[str], names: dict, highlight: int | None = None,
              height: int = 230) -> go.Figure:
    cmap = color_map(classes)
    labels = [names.get(c, c) for c in classes]
    colors = [cmap[c] if (highlight is None or i == highlight) else "#C9D3DE" for i, c in enumerate(classes)]
    fig = go.Figure(go.Bar(x=probs, y=labels, orientation="h", marker_color=colors,
                           text=[f"{p:.1%}" for p in probs], textposition="outside",
                           textfont=dict(family=MONO, size=11, color=INK), cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", showgrid=True)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig = base_layout(fig, height=height, legend=False)
    fig.update_layout(margin=dict(l=8, r=24, t=8, b=28))
    return fig


# ---- static export (Matplotlib) -----------------------------------------------------------
def export_static_charts(out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#C9D0D9", "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID})
    saved = []
    hist = load_json(HISTORY_PATH)
    if hist:
        h = hist["history"]; ep = [r["epoch"] for r in h]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ep, [r["train_loss"] for r in h], "-o", ms=3, color=BLUE, label="Training")
        ax[0].plot(ep, [r["val_loss"] for r in h], "--o", ms=3, color=TEAL, label="Validation")
        ax[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy"); ax[0].legend(frameon=False)
        ax[1].plot(ep, [r["train_acc"] for r in h], "-o", ms=3, color=BLUE, label="Training (clip)")
        ax[1].plot(ep, [r["val_clip_acc"] for r in h], "--o", ms=3, color=TEAL, label="Validation (clip)")
        ax[1].plot(ep, [r["val_video_acc"] for r in h], ":", color=AMBER, label="Validation (video)")
        ax[1].set(title="Accuracy", xlabel="Epoch", ylim=(0, 1.02)); ax[1].legend(frameon=False)
        fig.tight_layout(); p = out_dir / "training_curves.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)
    m = load_json(TEST_METRICS_PATH)
    if m:
        cm = np.array(m["confusion_matrix"]); n = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(m["classes"])), m["classes"], rotation=35, ha="right")
        ax.set_yticks(range(len(m["classes"])), m["classes"]); ax.grid(False)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if n[i, j] > 0.5 else INK, fontsize=10)
        ax.set(xlabel="Predicted", ylabel="True", title=f"Confusion matrix — test split (acc {m['accuracy']:.1%})")
        fig.tight_layout(); p = out_dir / "confusion_matrix.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)
    d = load_json(DISTRIBUTION_PATH)
    if d:
        cls = list(d["counts"]); left = np.zeros(len(cls))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for s, col in (("train", BLUE), ("val", TEAL), ("test", AMBER)):
            v = np.array([d["counts"][c].get(s, 0) for c in cls]); ax.barh(cls, v, left=left, color=col, label=s); left += v
        ax.invert_yaxis(); ax.set(xlabel="Videos", title="Class distribution"); ax.legend(frameon=False)
        fig.tight_layout(); p = out_dir / "class_distribution.png"; fig.savefig(p, dpi=160); plt.close(fig); saved.append(p)
    return saved
