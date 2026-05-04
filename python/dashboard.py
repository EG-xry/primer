#!/usr/bin/env python3
"""
视网膜分析 Dashboard — 静态 HTML 报告生成器
============================================
读取运行日志和输出图表，生成一个自包含的 HTML dashboard。
"""
import base64, glob, os, re
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / "output"
LOG_FILES = sorted(OUTPUT_DIR.glob("run*.log"), key=os.path.getmtime, reverse=True)
IMAGES = sorted(OUTPUT_DIR.glob("fig_*.png"))

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def parse_log(path):
    with open(path, "r") as f:
        return f.read()

# 选最新日志
log_text = parse_log(LOG_FILES[0]) if LOG_FILES else "无日志文件"
log_name = LOG_FILES[0].name if LOG_FILES else "N/A"

# 提取时间统计
timing_lines = [l for l in log_text.split("\n") if "⏱" in l or "总计" in l or "█" in l]
timing_block = "\n".join(timing_lines) if timing_lines else "未找到时间统计"

# 读取 timing.json (如果存在)
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
    timing_chart_html = f'<div style="margin:16px 0">{bars}<div style="margin-top:8px;font-weight:600;font-size:15px">总计: {total:.1f}s | MPS: {"✅" if timing_json.get("mps_available") else "❌"} | {timing_json.get("timestamp","")[:19]}</div></div>'

# 提取参数
params_lines = [l for l in log_text.split("\n") if any(k in l for k in ["项目根目录", "数据目录", "输出目录", "分析细胞", "总帧数", "空间维度", "N=", "参数量", "脉冲率", "MPS"])]
params_block = "\n".join(params_lines[:20])

# 构建 HTML
imgs_html = ""
for img in IMAGES:
    b64 = img_to_b64(img)
    imgs_html += f'<div class="card"><h3>{img.stem}</h3><img src="data:image/png;base64,{b64}"></div>\n'

html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Retina Analysis Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #f5f5f7; color: #333; }}
.header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 40px; }}
.header h1 {{ margin: 0; font-size: 28px; }}
.header p {{ margin: 5px 0 0; opacity: 0.8; }}
.container {{ max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
.section {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.section h2 {{ margin-top: 0; color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 8px; }}
pre {{ background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.5; max-height: 400px; overflow-y: auto; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
.card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card h3 {{ margin: 0 0 12px; color: #555; font-size: 14px; }}
.card img {{ width: 100%; border-radius: 8px; }}
.badge {{ display: inline-block; background: #667eea; color: white; padding: 2px 10px; border-radius: 12px; font-size: 12px; margin-right: 6px; }}
.meta {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 12px 0; }}
.meta-item {{ background: #f0f0f5; padding: 12px 16px; border-radius: 8px; flex: 1; min-width: 200px; }}
.meta-item .label {{ font-size: 12px; color: #888; }}
.meta-item .value {{ font-size: 18px; font-weight: 600; color: #333; }}
</style>
</head>
<body>
<div class="header">
  <h1>🧠 Retina Spike Train Analysis Dashboard</h1>
  <p>Analysis of Neuronal Spike Trains, Deconstructed — Python 复现 (MPS GPU 优化)</p>
</div>
<div class="container">

<div class="section">
  <h2>📋 运行概要</h2>
  <div class="meta">
    <div class="meta-item"><div class="label">日志文件</div><div class="value">{log_name}</div></div>
    <div class="meta-item"><div class="label">生成时间</div><div class="value">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div></div>
    <div class="meta-item"><div class="label">输出图表</div><div class="value">{len(IMAGES)} 张</div></div>
    <div class="meta-item"><div class="label">加速方案</div><div class="value"><span class="badge">MPS GPU</span><span class="badge">NumPy 优化</span></div></div>
  </div>
</div>

<div class="section">
  <h2>⚙️ 数据参数</h2>
  <pre>{params_block}</pre>
</div>

<div class="section">
  <h2>⏱ 运行时间统计</h2>
  {timing_chart_html}
  <pre>{timing_block}</pre>
</div>

<div class="section">
  <h2>🖼 分析图表</h2>
  <div class="grid">
    {imgs_html}
  </div>
</div>

<div class="section">
  <h2>📝 完整运行日志</h2>
  <pre>{log_text[-5000:]}</pre>
</div>

</div>
</body>
</html>"""

out_path = OUTPUT_DIR / "dashboard.html"
with open(out_path, "w") as f:
    f.write(html)

print(f"Dashboard 已生成: {out_path}")
