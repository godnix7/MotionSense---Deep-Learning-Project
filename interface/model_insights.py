"""PAGE 4 — Model Insights. Every number on this page is read from files written by
training/train.py and training/evaluate.py."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from interface.diagram import architecture_svg
from interface.ui import experiment_files, model_meta, page_header, section, strip
from utils.config import CLASSES, DISPLAY_NAMES, REPORTS_DIR
from utils.visualization import (accuracy_curve, confusion_matrix_fig, distribution_fig, load_json, loss_curve,
                                 per_class_fig)


def _missing(what: str, cmd: str):
    st.markdown(f'<div class="ms-panel ms-note">{what} not available yet. Generate it with '
                f'<span class="ms-mono">{cmd}</span>.</div>', unsafe_allow_html=True)


def render():
    page_header("Model insights", "Training & Evaluation Results",
                "Learning curves from the training run and final metrics on the held-out official UCF101 test split.")
    meta = model_meta()
    hist, test, dist = experiment_files()
    classes = (test or {}).get("classes") or meta.get("classes") or CLASSES
    names = {c: DISPLAY_NAMES.get(c, c) for c in classes}
    tc = (hist or {}).get("config") or meta.get("train_config", {})

    # ---- headline metrics ---------------------------------------------------------------
    if test:
        st.markdown(strip([
            ("Test accuracy", f"{test['accuracy']:.1%}", "", f"video-level · {test['n_samples']} test videos"),
            ("Clip accuracy", f"{test['clip_accuracy']:.1%}", "", f"single {tc.get('seq_len', 16)}-frame windows"),
            ("Macro precision", f"{test['precision_macro']:.3f}", "", ""),
            ("Macro recall", f"{test['recall_macro']:.3f}", "", ""),
            ("Macro F1", f"{test['f1_macro']:.3f}", "", test.get("split", "")),
        ]), unsafe_allow_html=True)
        st.markdown('<div class="ms-note" style="margin-top:8px">Offline test performance (above) is measured on '
                    'UCF101 test videos never seen in training or model selection. Live webcam behaviour differs '
                    '(viewpoint, lighting, clothing, framing) and is reported separately on the Live page.</div>',
                    unsafe_allow_html=True)
    else:
        _missing("Test metrics", "python -m training.evaluate")

    section("01", "Architecture", "MobileNetV3-Large → LSTM → FC → Softmax")
    st.markdown(architecture_svg(tc.get("seq_len", 16), tc.get("img_size", 160), tc.get("sample_fps", 8.0),
                                 tc.get("lstm_hidden", 256), len(classes), compact=True), unsafe_allow_html=True)

    # ---- learning curves ------------------------------------------------------------------
    section("02", "Learning curves", f"{len(hist['history'])} epochs · best val macro-F1 {hist['best_val_f1_macro']:.3f}"
            if hist else "")
    if hist:
        c1, c2 = st.columns(2, gap="large")
        c1.plotly_chart(loss_curve(hist["history"]), width="stretch", config={"displayModeBar": False})
        c2.plotly_chart(accuracy_curve(hist["history"]), width="stretch", config={"displayModeBar": False})
        st.markdown(f'<div class="ms-note">Training metrics use augmented clips (label smoothing 0.1), so training loss '
                    f'is not directly comparable to validation loss. Validation = training-list groups '
                    f'{", ".join(f"g{g:02d}" for g in (dist or {}).get("val_groups", []))} (never in test). '
                    f'Total training time {hist["total_time_s"] / 60:.1f} min on {hist["device"]}.</div>',
                    unsafe_allow_html=True)
    else:
        _missing("Training history", "python -m training.train")

    # ---- confusion + report ---------------------------------------------------------------
    section("03", "Test-set evaluation", "official split — confusion matrix and per-class report")
    if test:
        c1, c2 = st.columns([1.05, 1], gap="large")
        disp = [names[c] for c in classes]
        c1.plotly_chart(confusion_matrix_fig(test["confusion_matrix"], disp), width="stretch",
                        config={"displayModeBar": False})
        with c2:
            rep = test["classification_report"]
            df = pd.DataFrame([{"Class": names[c], "Precision": rep[c]["precision"], "Recall": rep[c]["recall"],
                                "F1-score": rep[c]["f1-score"], "Support": int(rep[c]["support"])} for c in classes]
                              + [{"Class": "Macro avg", "Precision": rep["macro avg"]["precision"],
                                  "Recall": rep["macro avg"]["recall"], "F1-score": rep["macro avg"]["f1-score"],
                                  "Support": int(rep["macro avg"]["support"])},
                                 {"Class": "Weighted avg", "Precision": rep["weighted avg"]["precision"],
                                  "Recall": rep["weighted avg"]["recall"], "F1-score": rep["weighted avg"]["f1-score"],
                                  "Support": int(rep["weighted avg"]["support"])}])
            st.markdown('<div class="ms-label">Classification report</div>', unsafe_allow_html=True)
            st.dataframe(df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}),
                         hide_index=True, width="stretch")
            st.plotly_chart(per_class_fig(rep, classes, names), width="stretch", config={"displayModeBar": False})
    else:
        _missing("Confusion matrix and classification report", "python -m training.evaluate")

    # ---- data distribution + config -----------------------------------------------------
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        section("04", "Dataset class distribution", "videos per subset")
        if dist:
            st.plotly_chart(distribution_fig(dist, names), width="stretch", config={"displayModeBar": False})
        else:
            _missing("Class distribution", "python -m data.preprocess")
    with c2:
        section("05", "Training configuration")
        if tc:
            rows = [("Optimizer", f"AdamW (wd {tc.get('weight_decay')})"),
                    ("Learning rate", f"head {tc.get('lr_head')} · backbone {tc.get('lr_backbone')} · cosine"),
                    ("Loss", "CrossEntropyLoss (label smoothing 0.1)"),
                    ("Epochs / batch", f"{tc.get('epochs')} / {tc.get('batch_size')} clips"),
                    ("Sequence", f"{tc.get('seq_len')} frames @ {tc.get('sample_fps')} fps, {tc.get('img_size')}²"),
                    ("Backbone", "ImageNet MobileNetV3-L, frozen feature extractor" if not tc.get("unfreeze_blocks")
                     else f"ImageNet MobileNetV3-L, last {tc.get('unfreeze_blocks')} blocks fine-tuned"),
                    ("Temporal", f"LSTM hidden {tc.get('lstm_hidden')}, {tc.get('lstm_layers')} layer(s)"
                     + (", input [fₜ ; fₜ − fₜ₋₁]" if tc.get("use_deltas") else "")),
                    ("Augmentation", "resized crop, flip, brightness/contrast, grayscale, temporal stride jitter (GPU)"),
                    ("Precision", "mixed (FP16 autocast)" if tc.get("amp") else "FP32"),
                    ("Seed", str(tc.get("seed")))]
            st.dataframe(pd.DataFrame(rows, columns=["Setting", "Value"]), hide_index=True, width="stretch")

    sel = load_json(REPORTS_DIR / "model_selection.json")
    if sel:
        section("06", "Model selection", "chosen by validation macro-F1 only — the test split was used once, afterwards")
        best = max(sel, key=lambda k: sel[k]["best_val_f1_macro"])
        df = pd.DataFrame([{"Experiment": k, "Backbone": "frozen" if v["unfreeze_blocks"] == 0 else f"last {v['unfreeze_blocks']} blocks tuned",
                            "LSTM input": "[fₜ ; Δfₜ]" if v["use_deltas"] else "fₜ",
                            "Best val macro-F1": round(v["best_val_f1_macro"], 3),
                            "Selected": "yes" if k == best else ""} for k, v in sel.items()])
        st.dataframe(df, hide_index=True, width="stretch")
        st.markdown('<div class="ms-note">Validation errors of the first model clustered by UCF101 group (shared actor and '
                    'background), indicating scene memorisation. Feeding temporal feature differences and freezing the '
                    'backbone reduced this; the frozen variant was selected (best validation F1, fewest trainable parameters).</div>',
                    unsafe_allow_html=True)
