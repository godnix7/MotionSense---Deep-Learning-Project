"""Technical SVG architecture diagram (rendered inline; scales with container width)."""
from __future__ import annotations


def architecture_svg(seq_len: int = 16, img: int = 160, fps: float = 8.0, hidden: int = 256,
                     n_cls: int = 6, compact: bool = False) -> str:
    stages = [
        ("01", "VIDEO INPUT", "webcam · mp4 / avi / mov", "H×W×3 @ 25–60 fps"),
        ("02", "FRAME SAMPLING", f"fixed {fps:g} fps · rolling", f"T = {seq_len} ≈ {seq_len / fps:.0f} s"),
        ("03", "PREPROCESSING", "crop · ImageNet norm", f"T×3×{img}×{img}"),
        ("04", "MOBILENETV3-L", "ImageNet weights · per frame", "T×960 features"),
        ("05", "LSTM", f"[fₜ ; Δfₜ] · hidden {hidden}", "last ‖ mean"),
        ("06", "FC CLASSIFIER", f"{2 * hidden}→{hidden}→{n_cls}", f"{n_cls} logits"),
        ("07", "SOFTMAX", "EMA · threshold", "label + confidence"),
    ]
    n = len(stages)
    W, bw, gap, top = 1400, 180, 24, 34
    bh = 118
    x0 = (W - (n * bw + (n - 1) * gap)) / 2
    H = top + bh + (66 if not compact else 34)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" '
             'style="font-family:Inter,Helvetica Neue,Arial,sans-serif;display:block">',
             '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
             '<path d="M0,0 L10,5 L0,10 z" fill="#687586"/></marker>'
             '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
             '<line x1="0" y1="0" x2="0" y2="6" stroke="#DCE6F0" stroke-width="2"/></pattern></defs>']
    # bracket for learned components
    lx0 = x0 + 3 * (bw + gap) - 8
    lx1 = x0 + 5 * (bw + gap) + bw + 8
    parts.append(f'<rect x="{lx0}" y="{top - 22}" width="{lx1 - lx0}" height="{bh + 30}" fill="url(#hatch)" '
                 'stroke="#1769AA" stroke-dasharray="4 4" stroke-width="1" rx="2"/>')
    parts.append(f'<text x="{lx0 + 8}" y="{top - 8}" font-family="IBM Plex Mono,monospace" font-size="12" fill="#1769AA" '
                 'letter-spacing="1.5">TRAINABLE NETWORK · PyTorch</text>')
    for i, (idx, title, desc, shape) in enumerate(stages):
        x = x0 + i * (bw + gap)
        learned = 3 <= i <= 5
        fill = "#14263D" if learned else "#FFFFFF"
        ink = "#FFFFFF" if learned else "#202B38"
        sub = "#9FB3C8" if learned else "#687586"
        stroke = "#14263D" if learned else "#C9D2DC"
        parts.append(f'<rect x="{x}" y="{top}" width="{bw}" height="{bh - 12}" fill="{fill}" stroke="{stroke}" rx="2"/>')
        parts.append(f'<text x="{x + 12}" y="{top + 22}" font-family="IBM Plex Mono,monospace" font-size="11" '
                     f'fill="{"#7FD3CF" if learned else "#16A6A1"}" letter-spacing="1">{idx}</text>')
        parts.append(f'<text x="{x + 12}" y="{top + 46}" font-size="14" font-weight="700" fill="{ink}" letter-spacing=".3">{title}</text>')
        parts.append(f'<text x="{x + 12}" y="{top + 66}" font-size="11.5" fill="{sub}">{desc}</text>')
        parts.append(f'<line x1="{x + 12}" y1="{top + 78}" x2="{x + bw - 12}" y2="{top + 78}" stroke="{"#2C4361" if learned else "#E3E7ED"}"/>')
        parts.append(f'<text x="{x + 12}" y="{top + 96}" font-family="IBM Plex Mono,monospace" font-size="11" fill="{ink}">{shape}</text>')
        if i < n - 1:
            ay = top + (bh - 12) / 2
            parts.append(f'<line x1="{x + bw + 3}" y1="{ay}" x2="{x + bw + gap - 3}" y2="{ay}" stroke="#687586" '
                         'stroke-width="1.5" marker-end="url(#ah)"/>')
    if not compact:
        y = top + bh + 30
        for i, line in enumerate((
                "LIVE MODE  each sampled frame passes the CNN once; its feature is cached in a rolling buffer and the LSTM",
                "           re-classifies the latest window every 2 new frames.   Δfₜ = fₜ − fₜ₋₁ emphasises motion over static background.")):
            parts.append(f'<text x="{x0}" y="{y + i * 17}" font-family="IBM Plex Mono,monospace" font-size="12" '
                         f'fill="#687586" xml:space="preserve">{line}</text>')
    parts.append("</svg>")
    return "".join(parts)
