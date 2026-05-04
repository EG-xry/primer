#!/usr/bin/env python3
"""
Retina Analysis Dashboard — Static HTML Report Generator
=========================================================
Reads run logs and output charts, generates a self-contained HTML dashboard.

Usage:
    python dashboard.py                     # Use default theme (modern_purple)
    python dashboard.py --theme dark_neuron # Use dark neuron theme

Available themes:
    modern_purple  — Purple gradient (default)
    dark_neuron    — Dark neuroscience theme
    ocean_breeze   — Fresh ocean blue theme
    warm_earth     — Warm earth tone theme
    minimal_light  — Minimalist white theme
"""
import argparse, base64, glob, os, re
from pathlib import Path
from datetime import datetime

THEMES_DIR = Path(__file__).parent / "themes"
OUTPUT_DIR = Path(__file__).parent / "output"
AVAILABLE_THEMES = [f.stem for f in sorted(THEMES_DIR.glob("*.css"))] if THEMES_DIR.exists() else ["modern_purple"]

def load_theme_css(theme_name):
    """Load CSS from theme file, fallback to default inline CSS."""
    css_path = THEMES_DIR / f"{theme_name}.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            return f.read()
    # Fallback default
    return """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f5f5f7; color: #333; }
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 40px; }
.header h1 { margin: 0; font-size: 28px; }
.header p { margin: 5px 0 0; opacity: 0.8; }
.container { max-width: 1200px; margin: 20px auto; padding: 0 20px; }
.section { background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.section h2 { margin-top: 0; color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 8px; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.5; max-height: 400px; overflow-y: auto; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }
.card { background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.card h3 { margin: 0 0 12px; color: #555; font-size: 14px; }
.card img { width: 100%; border-radius: 8px; }
.badge { display: inline-block; background: #667eea; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; margin-right: 6px; }
.meta { display: flex; gap: 20px; flex-wrap: wrap; margin: 12px 0; }
.meta-item { background: #f0f0f5; padding: 12px 16px; border-radius: 8px; flex: 1; min-width: 200px; }
.meta-item .label { font-size: 12px; color: #888; }
.meta-item .value { font-size: 18px; font-weight: 600; color: #333; }
"""

LOG_FILES = sorted(OUTPUT_DIR.glob("run*.log"), key=os.path.getmtime, reverse=True)
IMAGES = sorted(OUTPUT_DIR.glob("fig_*.png"))

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def parse_log(path):
    with open(path, "r") as f:
        return f.read()

def build_dashboard(theme_name="modern_purple"):
    # Load theme CSS
    css = load_theme_css(theme_name)

    # Select latest log
    log_text = parse_log(LOG_FILES[0]) if LOG_FILES else "No log files found"
    log_name = LOG_FILES[0].name if LOG_FILES else "N/A"

    # Extract timing stats
    timing_lines = [l for l in log_text.split("\n") if "⏱" in l or "总计" in l or "█" in l]
    timing_block = "\n".join(timing_lines) if timing_lines else "No timing statistics found"

    # Read timing.json (if exists)
    timing_json = None
    timing_chart_html = ""
    timing_json_path = OUTPUT_DIR / "timing.json"
    if timing_json_path.exists():
        import json
        with open(timing_json_path) as f:
            timing_json = json.load(f)
        stages = timing_json.get("stages", {})
        total = timing_json.get("total", 1)
        colors_chart = ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#00f2fe", "#43e97b"]
        bars = ""
        for i, (name, sec) in enumerate(stages.items()):
            pct = sec / total * 100
            c = colors_chart[i % len(colors_chart)]
            bars += f'<div style="display:flex;align-items:center;margin:6px 0"><span style="width:160px;font-size:13px">{name}</span><div style="background:{c};height:28px;width:{pct*3}px;border-radius:4px;display:flex;align-items:center;padding:0 8px;color:white;font-size:12px;font-weight:600">{sec:.1f}s ({pct:.0f}%)</div></div>\n'
        timing_chart_html = f'<div style="margin:16px 0">{bars}<div style="margin-top:8px;font-weight:600;font-size:15px">Total: {total:.1f}s | MPS: {"✅" if timing_json.get("mps_available") else "❌"} | {timing_json.get("timestamp","")[:19]}</div></div>'

    # Extract parameters
    params_lines = [l for l in log_text.split("\n") if any(k in l for k in ["项目根目录", "数据目录", "输出目录", "分析细胞", "总帧数", "空间维度", "N=", "参数量", "脉冲率", "MPS", "root", "data", "output", "cell", "frames", "dimension"])]
    params_block = "\n".join(params_lines[:20])

    # Build image HTML
    imgs_html = ""
    for img in IMAGES:
        b64 = img_to_b64(img)
        imgs_html += f'<div class="card"><h3>{img.stem}</h3><img src="data:image/png;base64,{b64}"></div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Retina Analysis Dashboard — {theme_name}</title>
<style>
{css}
</style>
</head>
<body>
<div class="header">
  <h1>🧠 Retina Spike Train Analysis Dashboard</h1>
  <p>Analysis of Neuronal Spike Trains, Deconstructed — Python Reproduction (MPS GPU Optimized) | Theme: {theme_name}</p>
</div>
<div class="container">

<div class="section">
  <h2>📋 Run Summary</h2>
  <div class="meta">
    <div class="meta-item"><div class="label">Log File</div><div class="value">{log_name}</div></div>
    <div class="meta-item"><div class="label">Generated</div><div class="value">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div></div>
    <div class="meta-item"><div class="label">Output Charts</div><div class="value">{len(IMAGES)} figures</div></div>
    <div class="meta-item"><div class="label">Acceleration</div><div class="value"><span class="badge">MPS GPU</span><span class="badge">NumPy Optimized</span></div></div>
  </div>
</div>

<div class="section">
  <h2>⚙️ Data Parameters</h2>
  <pre>{params_block}</pre>
</div>

<div class="section">
  <h2>⏱ Runtime Statistics</h2>
  {timing_chart_html}
  <pre>{timing_block}</pre>
</div>

<div class="section">
  <h2>🖼 Analysis Charts</h2>
  <div class="grid">
    {imgs_html}
  </div>
</div>

<div class="section">
  <h2>📝 Full Run Log</h2>
  <pre>{log_text[-5000:]}</pre>
</div>

</div>
</body>
</html>"""

    out_path = OUTPUT_DIR / "dashboard.html"
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Dashboard generated: {out_path}")
    print(f"Theme: {theme_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Retina Analysis Dashboard")
    parser.add_argument("--theme", "-t", default="modern_purple",
                        choices=AVAILABLE_THEMES,
                        help=f"CSS theme to use (available: {', '.join(AVAILABLE_THEMES)})")
    parser.add_argument("--list-themes", action="store_true",
                        help="List available themes and exit")
    args = parser.parse_args()

    if args.list_themes:
        print("Available themes:")
        for t in AVAILABLE_THEMES:
            css_path = THEMES_DIR / f"{t}.css"
            with open(css_path) as f:
                first_line = f.readline().strip()
            desc = first_line.replace("/*", "").replace("*/", "").strip()
            print(f"  {t:20s} — {desc}")
    else:
        build_dashboard(args.theme)
