#!/usr/bin/env python3
"""
Retina Analysis Dashboard — Static HTML Report Generator
=========================================================
Reads run logs and output charts, generates a self-contained HTML dashboard.

Usage:
    python dashboard.py                     # Generate both EN and ZH dashboards (default theme)
    python dashboard.py --theme dark_neuron # Use dark neuron theme
    python dashboard.py --lang en           # Generate English only
    python dashboard.py --lang zh           # Generate Chinese only
    python dashboard.py --lang both         # Generate both (default)

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

# ── Stage name translations (timing.json keys are in Chinese) ──────────────────
STAGE_NAME_EN = {
    "阶段一: 数据预处理":  "Stage 1: Data Preprocessing",
    "阶段二: STA":        "Stage 2: STA",
    "阶段三: STC":        "Stage 3: STC",
    "阶段四: MNE":        "Stage 4: MNE",
    "阶段五: GLM":        "Stage 5: GLM",
    "阶段六: 预测验证":    "Stage 6: Prediction Validation",
    "阶段七: 图表生成":    "Stage 7: Chart Generation",
}

# ── Log text translations (Chinese → English) for timing/log blocks ────────────
LOG_TEXT_TRANSLATIONS = {
    # ── Section headers ────────────────────────────────────────────────────────
    "视网膜模块代码复现 - Python 版本":  "Retina Module Reproduction — Python Version",
    "阶段一: 数据准备与预处理":          "Stage 1: Data Preparation & Preprocessing",
    "阶段二: STA 脉冲触发平均":          "Stage 2: STA Spike-Triggered Average",
    "阶段三: STC 脉冲触发协方差 [优化版]": "Stage 3: STC Spike-Triggered Covariance [Optimized]",
    "阶段四: MNE 最大噪声熵":            "Stage 4: MNE Maximally Non-informative Encoding",
    "阶段五: GLM 广义线性模型 [MPS 优化版]": "Stage 5: GLM Generalized Linear Model [MPS Optimized]",
    "阶段六: 预测与验证":                "Stage 6: Prediction & Validation",
    "阶段七: 生成图表":                  "Stage 7: Chart Generation",
    # ── Timing lines (summary table) ──────────────────────────────────────────
    "阶段一: 数据预处理":  "Stage 1: Data Preprocessing",
    "阶段二: STA":        "Stage 2: STA",
    "阶段三: STC":        "Stage 3: STC",
    "阶段四: MNE":        "Stage 4: MNE",
    "阶段五: GLM":        "Stage 5: GLM",
    "阶段六: 预测验证":    "Stage 6: Prediction Validation",
    "阶段七: 图表生成":    "Stage 7: Chart Generation",
    # ── Elapsed times ─────────────────────────────────────────────────────────
    "⏱ 阶段一耗时":    "⏱ Stage 1 elapsed",
    "⏱ 阶段二耗时":    "⏱ Stage 2 elapsed",
    "⏱ 阶段三耗时":    "⏱ Stage 3 elapsed",
    "⏱ 阶段四耗时":    "⏱ Stage 4 elapsed",
    "⏱ 阶段五耗时":    "⏱ Stage 5 elapsed",
    "⏱ 阶段六耗时":    "⏱ Stage 6 elapsed",
    "⏱ 阶段七耗时":    "⏱ Stage 7 elapsed",
    # ── Completion markers ────────────────────────────────────────────────────
    "✅ 所有阶段完成!":   "✅ All stages completed!",
    "✓ 阶段一完成: 数据预处理就绪":   "✓ Stage 1 completed: Data preprocessing ready",
    "✓ 阶段二完成: STA 模型就绪":     "✓ Stage 2 completed: STA model ready",
    "✓ 阶段三完成: STC 分析就绪":     "✓ Stage 3 completed: STC analysis ready",
    "✓ 阶段四完成: MNE 模型就绪":     "✓ Stage 4 completed: MNE model ready",
    "✓ 阶段五完成: GLM 模型就绪":     "✓ Stage 5 completed: GLM model ready",
    "✓ 阶段六完成: 预测与验证就绪":    "✓ Stage 6 completed: Prediction & validation ready",
    "✓ 阶段七完成: 所有图表已保存到":  "✓ Stage 7 completed: All charts saved to",
    "✓ 阶段一完成":      "✓ Stage 1 completed",
    "✓ 阶段二完成":      "✓ Stage 2 completed",
    "✓ 阶段三完成":      "✓ Stage 3 completed",
    "✓ 阶段四完成":      "✓ Stage 4 completed",
    "✓ 阶段五完成":      "✓ Stage 5 completed",
    "✓ 阶段六完成":      "✓ Stage 6 completed",
    "✓ 阶段七完成":      "✓ Stage 7 completed",
    # ── Config / parameters ───────────────────────────────────────────────────
    "项目根目录":        "Project root",
    "数据目录":          "Data directory",
    "输出目录":          "Output directory",
    "使用数据目录":      "Using data directory",
    "分析细胞":          "Analysis cells",
    "总帧数":            "Total frames",
    "空间维度":          "Spatial dimension",
    "时间深度":          "Temporal depth",
    "延迟":              "Delay",
    "训练帧数":          "Training frames",
    "参数量":            "Parameters",
    "脉冲率":            "Spike rate",
    "总计":              "Total",
    # ── Process details ───────────────────────────────────────────────────────
    "处理刺激配置":      "Processing stimulus config",
    "完成":              "Done",
    "计算完成":          "computation done",
    "折":                "folds",
    "检测到 Apple MPS GPU，将使用 GPU 加速": "Apple MPS GPU detected, GPU acceleration enabled",
    "使用 Apple MPS GPU 加速":               "Using Apple MPS GPU acceleration",
    "先验协方差":        "Prior covariance",
    "脉冲触发协方差":    "Spike-triggered covariance",
    "特征分解":          "Eigendecomposition",
    "范围":              "Range",
    "零分布":            "Null distribution",
    "次置换":            "permutations",
    "置换":              "Permutation",
    "耗时":              "elapsed",
    "剩余":              "remaining",
    "MPS 零分布完成":    "MPS null distribution done",
    "总耗时":            "total elapsed",
    "个显著 STC 特征":   "significant STC features",
    "零分布范围":        "Null distribution range",
    "广义线性模型":       "Generalized Linear Model",
    "MPS 优化版":        "MPS Optimized",
    "模型就绪":          "model ready",
    "所有图表已保存到":   "All charts saved to",
    # ── Timing summary ────────────────────────────────────────────────────────
    "📊 运行时间统计":    "📊 Runtime Statistics",
    "⏱ 时间统计已保存":   "⏱ Timing statistics saved",
    "输出图表保存在":     "Output charts saved in",
    "包含":               "Including",
    # ── MNE / GLM details ─────────────────────────────────────────────────────
    "MNE 迭代优化":      "MNE iterative optimization",
    "初始化":            "Initialization",
    "迭代":              "Iteration",
    "收敛":              "Converged",
    "似然":              "Likelihood",
    "平均对数似然差":     "Mean log-likelihood difference",
    "步骤":              "Step",
    # ── Additional log phrases ────────────────────────────────────────────────
    "使用Data directory":  "Using data directory",
    "使用数据":           "Using data",
    "使用 PyTorch MPS GPU 加速": "Using PyTorch MPS GPU acceleration",
    "优化器":             "Optimizer",
    "内部":               "Inner",
    "训练":               "Train",
    "测试":               "Test",
    "显著二次特征":       "Significant quadratic features",
    # ── GLM fitting phrases ───────────────────────────────────────────────────
    "拟合 Poisson GLM":   "Fitting Poisson GLM",
    "拟合":               "Fitting",
    "PCA 降维":           "PCA reduction",
    "维":                 "dims",
    "解释方差":           "Explained variance",
    "脉冲历史":           "Spike history",
    "参数":               "Params",
    "模型":               "Model",
    "平均对数Likelihood差": "Mean log-likelihood difference",
    "默认主题":           "Default theme",
    "紫色渐变":           "Purple gradient",
    "使用":               "Using",
}

def translate_log_text(text):
    """Translate Chinese log text to English by replacing known phrases."""
    for zh, en in LOG_TEXT_TRANSLATIONS.items():
        text = text.replace(zh, en)
    return text

# ── i18n translation dictionaries ──────────────────────────────────────────────
I18N = {
    "en": {
        "html_lang":          "en",
        "page_title":         "Retina Spike Train Analysis Dashboard",
        "page_subtitle":      "Analysis of Neuronal Spike Trains, Deconstructed — Python Reproduction (MPS GPU Optimized)",
        "section_summary":    "📋 Run Summary",
        "label_log_file":     "Log File",
        "label_generated":    "Generated",
        "label_output_charts":"Output Charts",
        "label_acceleration": "Acceleration",
        "section_params":     "⚙️ Data Parameters",
        "section_timing":     "⏱ Runtime Statistics",
        "section_charts":     "🖼 Analysis Charts",
        "section_log":        "📝 Full Run Log",
        "figures_unit":       "figures",
        "theme_label":        "Theme",
        "total_label":        "Total",
        "no_log":             "No log files found",
        "no_timing":          "No timing statistics found",
        "badge_gpu":          "MPS GPU",
        "badge_numpy":        "NumPy Optimized",
        "lang_btn":           "中文",
        "other_lang_file":    "dashboard_zh.html",
    },
    "zh": {
        "html_lang":          "zh-CN",
        "page_title":         "视网膜脉冲序列分析仪表板",
        "page_subtitle":      "神经元脉冲序列分析（解构版）— Python 复现（MPS GPU 优化）",
        "section_summary":    "📋 运行摘要",
        "label_log_file":     "日志文件",
        "label_generated":    "生成时间",
        "label_output_charts":"输出图表",
        "label_acceleration": "加速方式",
        "section_params":     "⚙️ 数据参数",
        "section_timing":     "⏱ 运行时间统计",
        "section_charts":     "🖼 分析图表",
        "section_log":        "📝 完整运行日志",
        "figures_unit":       "张图表",
        "theme_label":        "主题",
        "total_label":        "总计",
        "no_log":             "未找到日志文件",
        "no_timing":          "未找到运行时间统计",
        "badge_gpu":          "MPS GPU",
        "badge_numpy":        "NumPy 优化",
        "lang_btn":           "English",
        "other_lang_file":    "dashboard_en.html",
    },
}

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

LOG_FILES = sorted(OUTPUT_DIR.glob("run*.log"), key=os.path.getmtime, reverse=True) if OUTPUT_DIR.exists() else []
IMAGES = sorted(OUTPUT_DIR.glob("fig_*.png")) if OUTPUT_DIR.exists() else []

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def parse_log(path):
    with open(path, "r") as f:
        return f.read()

def build_dashboard(theme_name="modern_purple", lang="en"):
    """Build a static HTML dashboard in the specified language.

    Args:
        theme_name: CSS theme name
        lang: 'en' or 'zh'
    """
    t = I18N[lang]

    # Load theme CSS
    css = load_theme_css(theme_name)

    # Select latest log
    log_text = parse_log(LOG_FILES[0]) if LOG_FILES else t["no_log"]
    log_name = LOG_FILES[0].name if LOG_FILES else "N/A"

    # For English version, translate the log text
    if lang == "en":
        log_text_display = translate_log_text(log_text)
    else:
        log_text_display = log_text

    # Extract timing stats (from display text so language matches)
    timing_lines = [l for l in log_text_display.split("\n") if "⏱" in l or "总计" in l or "Total" in l or "█" in l]
    timing_block = "\n".join(timing_lines) if timing_lines else t["no_timing"]

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
            # Translate stage names for English
            display_name = STAGE_NAME_EN.get(name, name) if lang == "en" else name
            pct = sec / total * 100
            c = colors_chart[i % len(colors_chart)]
            bars += f'<div style="display:flex;align-items:center;margin:6px 0"><span style="width:220px;font-size:13px">{display_name}</span><div style="background:{c};height:28px;width:{pct*3}px;border-radius:4px;display:flex;align-items:center;padding:0 8px;color:white;font-size:12px;font-weight:600">{sec:.1f}s ({pct:.0f}%)</div></div>\n'
        timing_chart_html = f'<div style="margin:16px 0">{bars}<div style="margin-top:8px;font-weight:600;font-size:15px">{t["total_label"]}: {total:.1f}s | MPS: {"✅" if timing_json.get("mps_available") else "❌"} | {timing_json.get("timestamp","")[:19]}</div></div>'

    # Extract parameters (from display text so language matches)
    params_lines = [l for l in log_text_display.split("\n") if any(k in l for k in ["项目根目录", "数据目录", "输出目录", "分析细胞", "总帧数", "空间维度", "N=", "参数量", "脉冲率", "MPS", "Project root", "Data directory", "Output directory", "Analysis cells", "Total frames", "Spatial dimension", "Parameters", "Spike rate", "root", "data", "output", "cell", "frames", "dimension"])]
    params_block = "\n".join(params_lines[:20])

    # Build image HTML
    imgs_html = ""
    for img in IMAGES:
        b64 = img_to_b64(img)
        imgs_html += f'<div class="card"><h3>{img.stem}</h3><img src="data:image/png;base64,{b64}"></div>\n'

    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_images = len(IMAGES)

    html = f"""<!DOCTYPE html>
<html lang="{t['html_lang']}">
<head>
<meta charset="UTF-8">
<title>{t['page_title']} — {theme_name}</title>
<style>
{css}
.lang-btn {{ position: fixed; top: 16px; right: 24px; z-index: 9999; background: rgba(255,255,255,0.9); color: #333; border: 1px solid #ccc; border-radius: 8px; padding: 6px 18px; font-size: 14px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.10); transition: background 0.2s; text-decoration: none; }}
.lang-btn:hover {{ background: #667eea; color: white; }}
</style>
</head>
<body>
<a class="lang-btn" href="{t['other_lang_file']}">{t['lang_btn']}</a>
<div class="header">
  <h1>🧠 {t['page_title']}</h1>
  <p>{t['page_subtitle']} | {t['theme_label']}: {theme_name}</p>
</div>
<div class="container">

<div class="section">
  <h2>{t['section_summary']}</h2>
  <div class="meta">
    <div class="meta-item"><div class="label">{t['label_log_file']}</div><div class="value">{log_name}</div></div>
    <div class="meta-item"><div class="label">{t['label_generated']}</div><div class="value">{generated_time}</div></div>
    <div class="meta-item"><div class="label">{t['label_output_charts']}</div><div class="value">{num_images} {t['figures_unit']}</div></div>
    <div class="meta-item"><div class="label">{t['label_acceleration']}</div><div class="value"><span class="badge">{t['badge_gpu']}</span><span class="badge">{t['badge_numpy']}</span></div></div>
  </div>
</div>

<div class="section">
  <h2>{t['section_params']}</h2>
  <pre>{params_block}</pre>
</div>

<div class="section">
  <h2>{t['section_timing']}</h2>
  {timing_chart_html}
  <pre>{timing_block}</pre>
</div>

<div class="section">
  <h2>{t['section_charts']}</h2>
  <div class="grid">
    {imgs_html}
  </div>
</div>

<div class="section">
  <h2>{t['section_log']}</h2>
  <pre>{log_text_display[-5000:]}</pre>
</div>

</div>
</body>
</html>"""

    out_filename = f"dashboard_{lang}.html"
    out_path = OUTPUT_DIR / out_filename
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Dashboard generated: {out_path}  [{lang.upper()}]")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Retina Analysis Dashboard")
    parser.add_argument("--theme", "-t", default="modern_purple",
                        choices=AVAILABLE_THEMES,
                        help=f"CSS theme to use (available: {', '.join(AVAILABLE_THEMES)})")
    parser.add_argument("--lang", "-l", default="both",
                        choices=["en", "zh", "both"],
                        help="Language: en (English), zh (Chinese), both (default)")
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
        langs = ["en", "zh"] if args.lang == "both" else [args.lang]
        for lang in langs:
            build_dashboard(args.theme, lang)
        print(f"Theme: {args.theme}")
