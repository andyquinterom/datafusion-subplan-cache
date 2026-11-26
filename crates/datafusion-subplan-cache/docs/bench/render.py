#!/usr/bin/env python3
"""Render docs/bench/results.json (from `--example bench`) into SVG charts.

    python3 crates/datafusion-subplan-cache/docs/bench/render.py

No dependencies. The SVGs follow the viewer's light/dark preference.
"""
import json
import os
import sys
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
FORMATS = [("delta", "Delta Lake"), ("ducklake", "DuckLake")]

STYLE = """
<style>
  svg { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  .t1 { fill: #1f2328; font-size: 15px; font-weight: 600; }
  .t2 { fill: #1f2328; font-size: 13px; font-weight: 600; }
  .lab { fill: #1f2328; font-size: 12px; }
  .mut { fill: #59636e; font-size: 11px; }
  .val { fill: #59636e; font-size: 11px; font-variant-numeric: tabular-nums; }
  .x { fill: #2a78d6; font-size: 11px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .grid { stroke: #d1d9e0; stroke-width: 1; }
  .axis { stroke: #b7bfc7; stroke-width: 1; }
  .van { fill: #a3a29b; }
  .cac { fill: #2a78d6; }
  .vanl { stroke: #a3a29b; fill: none; stroke-width: 2; }
  .cacl { stroke: #2a78d6; fill: none; stroke-width: 2; }
  .vand { fill: #a3a29b; }
  .cacd { fill: #2a78d6; }
  @media (prefers-color-scheme: dark) {
    .t1, .t2, .lab { fill: #f0f6fc; }
    .mut, .val { fill: #9198a1; }
    .x { fill: #4d9bf0; }
    .grid { stroke: #2f353d; }
    .axis { stroke: #3d444d; }
    .van, .vand { fill: #75746e; }
    .cac, .cacd { fill: #3987e5; }
    .vanl { stroke: #75746e; }
    .cacl { stroke: #3987e5; }
  }
</style>
"""


def svg(w, h, body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img">'
        f"<title>{escape(title)}</title>{STYLE}{body}</svg>\n"
    )


def text(x, y, s, cls, anchor="start"):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(str(s))}</text>'


def rect(x, y, w, h, cls, tip):
    # 4px rounding at the data end only: draw a rounded rect plus a square cap at the baseline.
    w = max(w, 1.5)
    r = min(3, w / 2, h / 2)
    return (
        f'<g class="{cls}"><title>{escape(tip)}</title>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r:.1f}"/>'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{min(w, r + 0.5):.1f}" height="{h:.1f}"/></g>'
    )


def fmt_ms(v):
    if v >= 100:
        return f"{v:.0f} ms"
    if v >= 10:
        return f"{v:.1f} ms"
    return f"{v:.1f} ms"


def nice_max(v):
    for e in range(-1, 7):
        for m in [1, 2, 2.5, 5]:
            c = m * 10**e
            if c >= v:
                return c
    return v


def ticks(vmax, n=4):
    top = nice_max(vmax)
    step = top / n
    return top, [step * i for i in range(n + 1)]


def legend(x, y):
    return (
        f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" class="van"/>'
        + text(x + 15, y, "vanilla DataFusion", "lab")
        + f'<rect x="{x + 140}" y="{y - 9}" width="10" height="10" rx="2" class="cac"/>'
        + text(x + 155, y, "DataFusion + subplan cache", "lab")
    )


def bars_chart(rows_by_fmt, title, subtitle, note_fn, fname, label_w=260):
    """Horizontal paired bars, one panel per format, rows = steps."""
    labels = [r["label"] for r in rows_by_fmt[FORMATS[0][0]]]
    n = len(labels)
    panel_w = 290
    bar_h, gap_in, row_h = 9, 2, 34
    top = 100
    w = label_w + 2 * panel_w + 40
    h = top + n * row_h + 34
    out = [text(16, 26, title, "t1"), text(16, 46, subtitle, "mut"), legend(16, 66)]
    for pi, (fmt, fname_h) in enumerate(FORMATS):
        rows = rows_by_fmt[fmt]
        x0 = label_w + pi * (panel_w + 20)
        vmax = max(max(r["vanilla_ms"], r["cached_ms"]) for r in rows)
        tmax, tks = ticks(vmax * 1.32)
        sx = (panel_w - 10) / tmax
        out.append(text(x0, top - 2, fname_h, "t2"))
        for t in tks:
            gx = x0 + t * sx
            out.append(f'<line x1="{gx:.1f}" y1="{top + 6}" x2="{gx:.1f}" y2="{top + n * row_h + 4}" class="grid"/>')
            out.append(text(gx, top + n * row_h + 18, f"{t:g}", "mut", "middle"))
        out.append(text(x0 + panel_w - 10, top + n * row_h + 30, "ms (lower is better)", "mut", "end"))
        for i, r in enumerate(rows):
            y = top + 10 + i * row_h
            v, c = r["vanilla_ms"], r["cached_ms"]
            out.append(rect(x0, y, v * sx, bar_h, "van", f"{r['label']} — vanilla: {fmt_ms(v)}"))
            out.append(rect(x0, y + bar_h + gap_in, c * sx, bar_h, "cac", f"{r['label']} — cached: {fmt_ms(c)}"))
            out.append(text(x0 + v * sx + 5, y + 8, fmt_ms(v), "val"))
            sp = v / max(c, 0.01)
            sp_s = f"×{sp:.0f}" if sp >= 10 else f"×{sp:.1f}"
            label = f"{fmt_ms(c)}  {sp_s}"
            out.append(text(x0 + c * sx + 5, y + bar_h + gap_in + 8.5, label, "x" if sp >= 1.5 else "val"))
            note = note_fn(r)
            if note:
                out.append(text(x0 + c * sx + 5 + 6.3 * len(label) + 8, y + bar_h + gap_in + 8.5, note, "mut"))
    for i, lab in enumerate(labels):
        y = top + 10 + i * row_h
        out.append(text(16, y + 14, lab, "lab"))
    out.append(f'<line x1="{label_w - 8}" y1="{top + 6}" x2="{label_w - 8}" y2="{top + n * row_h + 4}" class="axis"/>')
    with open(os.path.join(OUT, fname), "w") as f:
        f.write(svg(w, h, "".join(out), title))


def line_panel(x0, y0, pw, ph, series, title, xlabels):
    """series: list of (cls, dotcls, name, values)."""
    out = [text(x0, y0 - 10, title, "t2")]
    vmax = max(max(vals) for _, _, _, vals in series)
    tmax, tks = ticks(vmax * 1.1)
    n = len(xlabels)
    sx = (pw - 70) / max(n - 1, 1)
    sy = ph / tmax
    for t in tks:
        gy = y0 + ph - t * sy
        out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x0 + pw - 70}" y2="{gy:.1f}" class="grid"/>')
        out.append(text(x0 - 6, gy + 4, f"{t:g}", "mut", "end"))
    for i, xl in enumerate(xlabels):
        if i % 2 == 0 or n <= 8:
            out.append(text(x0 + i * sx, y0 + ph + 16, xl, "mut", "middle"))
    for cls, dcls, name, vals in series:
        pts = " ".join(f"{x0 + i * sx:.1f},{y0 + ph - v * sy:.1f}" for i, v in enumerate(vals))
        out.append(f'<polyline points="{pts}" class="{cls}"/>')
        for i, v in enumerate(vals):
            out.append(
                f'<circle cx="{x0 + i * sx:.1f}" cy="{y0 + ph - v * sy:.1f}" r="3.2" class="{dcls}">'
                f"<title>{escape(name)} · {xlabels[i]}: {fmt_ms(v)}</title></circle>"
            )
        out.append(text(x0 + (n - 1) * sx + 8, y0 + ph - vals[-1] * sy + 4, fmt_ms(vals[-1]), "val"))
    return "".join(out)


def timeseries_chart(res):
    panels = [
        ("last_6h", "last 6 hours (ts >= now() - 6h)"),
        ("daily_all_time", "all time, per day"),
    ]
    pw, ph = 380, 150
    left, top = 56, 96
    w = left + 2 * (pw + 24)
    h = top + 2 * (ph + 70) - 10
    title = "Time series: a new hour of data lands, the dashboard refreshes"
    out = [
        text(16, 26, title, "t1"),
        text(16, 46, "One commit per hour (3 files). Query latency right after each commit; x = hours since the cache was empty.", "mut"),
        legend(16, 66),
    ]
    for ri, (panel, ptitle) in enumerate(panels):
        for ci, (fmt, fname) in enumerate(FORMATS):
            rows = sorted(
                [r for r in res if r["scenario"] == "timeseries" and r["format"] == fmt and r["panel"] == panel],
                key=lambda r: r["step"],
            )
            xl = ["cold"] + [f"+{r['step']}h" for r in rows[1:]]
            x0 = left + ci * (pw + 24)
            y0 = top + ri * (ph + 70)
            out.append(
                line_panel(
                    x0,
                    y0,
                    pw,
                    ph,
                    [
                        ("vanl", "vand", "vanilla", [r["vanilla_ms"] for r in rows]),
                        ("cacl", "cacd", "cached", [r["cached_ms"] for r in rows]),
                    ],
                    f"{fname} · {ptitle}",
                    xl,
                )
            )
    with open(os.path.join(OUT, "timeseries.svg"), "w") as f:
        f.write(svg(w, h, "".join(out), title))


def files_note(r):
    read = r["files_computed"] + r["files_edge"]
    if r["whole_hits"] and not read and not r["files_reused"]:
        return "whole result reused"
    return f"{read} of {r['files_total']} files read"


def main():
    doc = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.json")))
    res = doc["results"]

    def pick(scen):
        return {
            fmt: sorted([r for r in res if r["scenario"] == scen and r["format"] == fmt], key=lambda r: r["step"])
            for fmt, _ in FORMATS
        }

    s = doc["setup"]
    rows_m = s["rows_total"] / 1e6
    bars_chart(
        pick("cross_query"),
        "Different queries share one cache",
        f"Each query runs for the first time, after the ones above it. {rows_m:.0f}M rows, {s['hours'] * 3} files.",
        files_note,
        "cross_query.svg",
    )
    bars_chart(
        pick("maintenance"),
        "Writes and table maintenance: only what changed is read again",
        "The same all-time hourly rollup, run right after each operation.",
        files_note,
        "maintenance.svg",
    )
    timeseries_chart(res)
    print("wrote cross_query.svg, maintenance.svg, timeseries.svg")


if __name__ == "__main__":
    main()
