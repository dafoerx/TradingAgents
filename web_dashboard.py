"""
TradingAgents Web Dashboard
============================
Streamlit Web 仪表板 —— 支持查看历史分析记录 + 实时分析辩论进展。
需要 Streamlit >= 1.18 (Python >= 3.10)。

启动方式:
    streamlit run web_dashboard.py
"""

import streamlit as st
import json
import os
import sys
import glob
import subprocess
import time
import re
import hashlib
import threading
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (.env) for API keys
load_dotenv()

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TradingAgents Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

.debate-card {
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; border-left: 5px solid;
}
.bull-card { background: linear-gradient(135deg, #1a3a1a 0%, #0e1117 100%); border-left-color: #22c55e; }
.bear-card { background: linear-gradient(135deg, #3a1a1a 0%, #0e1117 100%); border-left-color: #ef4444; }
.judge-card { background: linear-gradient(135deg, #1a1a3a 0%, #0e1117 100%); border-left-color: #3b82f6; }
.aggressive-card { background: linear-gradient(135deg, #3a2a1a 0%, #0e1117 100%); border-left-color: #f97316; }
.conservative-card { background: linear-gradient(135deg, #1a2a3a 0%, #0e1117 100%); border-left-color: #06b6d4; }
.neutral-card { background: linear-gradient(135deg, #2a2a1a 0%, #0e1117 100%); border-left-color: #a855f7; }
.portfolio-card { background: linear-gradient(135deg, #1a2a2a 0%, #0e1117 100%); border-left-color: #10b981; }

.decision-badge {
    display: inline-block; padding: 0.3rem 1rem; border-radius: 20px;
    font-weight: 700; font-size: 1.2rem; margin: 0.5rem 0;
}
.badge-sell { background: #dc2626; color: white; }
.badge-buy { background: #16a34a; color: white; }
.badge-hold { background: #d97706; color: white; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%); }

.metric-card {
    background: #1e293b; border-radius: 12px; padding: 1rem 1.5rem;
    text-align: center; border: 1px solid #334155;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #38bdf8; }
.metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem; }

.stage-active { color: #38bdf8; font-weight: 700; }
.stage-done { color: #22c55e; }
.stage-pending { color: #64748b; }

.live-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0e1117 100%);
    border: 1px solid #38bdf8; border-radius: 12px; padding: 1rem 1.5rem;
    margin-bottom: 1rem;
}
.live-dot {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    background: #ef4444; margin-right: 8px; animation: blink 1s infinite;
}
@keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Chinese Translation Engine (LLM-powered)
# ─────────────────────────────────────────────
EN_ZH_MAP = {
    "BUY": "买入", "SELL": "卖出", "HOLD": "持有",
    "STRONG BUY": "强烈买入", "STRONG SELL": "强烈卖出",
}

# Translation cache directory
TRANSLATION_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".translation_cache")
os.makedirs(TRANSLATION_CACHE_DIR, exist_ok=True)

# In-memory cache + lock
_translation_mem_cache = {}
_translation_lock = threading.Lock()

# DeepSeek client (lazy init)
_llm_client = None


def _get_llm_client():
    """Lazy-initialize the DeepSeek (OpenAI-compatible) client."""
    global _llm_client
    if _llm_client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None
        _llm_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _llm_client


def _cache_key(text):
    """Generate a short hash key for translation caching."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_cached_translation(key):
    """Load translation from disk cache."""
    path = os.path.join(TRANSLATION_CACHE_DIR, key + ".txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError:
            pass
    return None


def _save_cached_translation(key, translated):
    """Save translation to disk cache."""
    path = os.path.join(TRANSLATION_CACHE_DIR, key + ".txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(translated)
    except IOError:
        pass


def _llm_translate(text, max_chars=4000):
    """Call DeepSeek API to translate English text to Chinese."""
    client = _get_llm_client()
    if client is None:
        return None

    # Truncate very long text to avoid excessive API cost
    truncated = text[:max_chars] if len(text) > max_chars else text

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一位专业的金融翻译员。请将以下英文金融分析报告/辩论内容翻译为地道的中文。\n"
                        "要求：\n"
                        "1. 保持原文的 Markdown 格式（标题、列表、加粗等）\n"
                        "2. 专业金融术语使用标准中文翻译\n"
                        "3. 股票代码、数字、日期等保持原样\n"
                        "4. 翻译要流畅自然，符合中文金融分析的表达习惯\n"
                        "5. 不要添加任何额外的解释或注释，只输出翻译结果"
                    ),
                },
                {
                    "role": "user",
                    "content": truncated,
                },
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("[WARN] LLM translation failed: %s" % e)
        return None


def translate_to_chinese(text):
    """Translate English text to Chinese using LLM with caching.

    Returns translated Chinese text. Falls back to original if translation fails.
    """
    if not text or not isinstance(text, str):
        return text or ""

    # Skip if already mostly Chinese
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars > len(text) * 0.3:
        return text

    # Very short text: quick keyword replacement only
    text_stripped = text.strip()
    if len(text_stripped) < 20:
        for en, zh in EN_ZH_MAP.items():
            if en.lower() in text_stripped.lower():
                return text_stripped.replace(en, zh).replace(en.lower(), zh)
        return text

    key = _cache_key(text_stripped)

    # Check in-memory cache
    with _translation_lock:
        if key in _translation_mem_cache:
            return _translation_mem_cache[key]

    # Check disk cache
    cached = _load_cached_translation(key)
    if cached:
        with _translation_lock:
            _translation_mem_cache[key] = cached
        return cached

    # Call LLM
    translated = _llm_translate(text_stripped)
    if translated:
        with _translation_lock:
            _translation_mem_cache[key] = translated
        _save_cached_translation(key, translated)
        return translated

    # Fallback: return original
    return text


# ─────────────────────────────────────────────
# Data Loading Utilities
# ─────────────────────────────────────────────
LIVE_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".live_state")


def get_live_state_path():
    return os.path.join(LIVE_STATE_DIR, "current_analysis.json")


def load_live_state():
    """Load the live analysis state from file."""
    filepath = get_live_state_path()
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def find_log_files(base_dir="eval_results"):
    """Scan eval_results for all JSON log files."""
    log_files = []
    pattern = os.path.join(base_dir, "**/full_states_log_*.json")
    for filepath in glob.glob(pattern, recursive=True):
        parts = Path(filepath).parts
        try:
            ticker_idx = parts.index("eval_results") + 1
            ticker = parts[ticker_idx] if ticker_idx < len(parts) else "Unknown"
        except (ValueError, IndexError):
            ticker = "Unknown"
        fname = Path(filepath).stem
        date_str = fname.replace("full_states_log_", "")
        log_files.append({
            "path": filepath,
            "ticker": ticker,
            "date": date_str,
            "label": "%s - %s" % (ticker, date_str),
        })

    results_pattern = os.path.join("results", "**/full_states_log_*.json")
    for filepath in glob.glob(results_pattern, recursive=True):
        parts = Path(filepath).parts
        ticker = parts[1] if len(parts) > 1 else "Unknown"
        fname = Path(filepath).stem
        date_str = fname.replace("full_states_log_", "")
        log_files.append({
            "path": filepath,
            "ticker": ticker,
            "date": date_str,
            "label": "%s - %s" % (ticker, date_str),
        })

    return sorted(log_files, key=lambda x: (x["ticker"], x["date"]), reverse=True)


def load_log_data(filepath):
    """Load and parse a JSON log file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        dates = list(data.keys())
        return data, dates
    return {}, []


def extract_decision(text):
    """Extract BUY/SELL/HOLD from text."""
    if not text:
        return "UNKNOWN"
    text_upper = text.upper()
    for keyword in ["STRONG BUY", "STRONG SELL", "SELL", "BUY", "HOLD"]:
        if keyword in text_upper:
            if "STRONG BUY" in text_upper:
                return "BUY"
            if "STRONG SELL" in text_upper:
                return "SELL"
            return keyword
    return "UNKNOWN"


def get_decision_badge(decision):
    decision = decision.upper().strip()
    zh = EN_ZH_MAP.get(decision, decision)
    if "SELL" in decision:
        return '<span class="decision-badge badge-sell">🔴 %s</span>' % zh
    elif "BUY" in decision:
        return '<span class="decision-badge badge-buy">🟢 %s</span>' % zh
    elif "HOLD" in decision:
        return '<span class="decision-badge badge-hold">🟡 %s</span>' % zh
    return '<span class="decision-badge" style="background:#475569;color:white;">%s</span>' % decision


# ─────────────────────────────────────────────
# Pipeline Stages (for progress display)
# ─────────────────────────────────────────────
PIPELINE_STAGES = [
    {"key": "starting", "name": "初始化", "icon": "🔧"},
    {"key": "market_report", "name": "市场分析师", "icon": "📈"},
    {"key": "sentiment_report", "name": "社交媒体分析师", "icon": "💬"},
    {"key": "news_report", "name": "新闻分析师", "icon": "📰"},
    {"key": "fundamentals_report", "name": "基本面分析师", "icon": "📊"},
    {"key": "investment_debate", "name": "投资辩论 (Bull vs Bear)", "icon": "⚔️"},
    {"key": "trader", "name": "交易员", "icon": "💰"},
    {"key": "risk_debate", "name": "风控辩论", "icon": "🛡️"},
    {"key": "final_decision", "name": "最终决策", "icon": "🎯"},
]


def render_pipeline_progress(current_stage):
    """Render a visual pipeline progress bar."""
    stage_keys = [s["key"] for s in PIPELINE_STAGES]
    current_idx = -1
    if current_stage in stage_keys:
        current_idx = stage_keys.index(current_stage)

    html_parts = ['<div style="display:flex;flex-wrap:wrap;gap:8px;margin:1rem 0;">']
    for i, stage in enumerate(PIPELINE_STAGES):
        if i < current_idx:
            css_class = "stage-done"
            prefix = "✅"
        elif i == current_idx:
            css_class = "stage-active"
            prefix = "🔄"
        else:
            css_class = "stage-pending"
            prefix = "⬜"
        html_parts.append(
            '<span class="%s" style="padding:4px 10px;border-radius:6px;'
            'background:#1e293b;border:1px solid #334155;font-size:0.85rem;">'
            '%s %s %s</span>' % (css_class, prefix, stage["icon"], stage["name"])
        )
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Display Components
# ─────────────────────────────────────────────
def render_analyst_report(title, icon, content):
    """Render an analyst report — default Chinese translation, original in toggle."""
    if not content:
        return
    translated = translate_to_chinese(content)
    decision = extract_decision(content[-500:] if len(content) > 500 else content)
    badge = get_decision_badge(decision)
    with st.expander("%s %s  %s" % (icon, title, badge), expanded=False):
        st.markdown(translated)
        with st.expander("📝 查看英文原文", expanded=False):
            st.markdown(content)


def render_debate_card(title, icon, content, card_class):
    """Render a debate participant card — default Chinese translation."""
    if not content:
        return
    translated = translate_to_chinese(content)
    decision = extract_decision(content[-500:] if len(content) > 500 else content)
    badge_html = get_decision_badge(decision)
    st.markdown(
        '<div class="debate-card %s"><h3>%s %s %s</h3></div>'
        % (card_class, icon, title, badge_html),
        unsafe_allow_html=True,
    )
    with st.expander("📄 查看 %s 完整论述（中文翻译）" % title, expanded=True):
        st.markdown(translated)
        with st.expander("📝 查看英文原文", expanded=False):
            st.markdown(content)


def render_overview_metrics(data):
    """Render overview metric cards."""
    cols = st.columns(4)
    ticker = data.get("company_of_interest", data.get("ticker", "N/A"))
    with cols[0]:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">%s</div>'
            '<div class="metric-label">股票代码</div></div>' % ticker,
            unsafe_allow_html=True,
        )
    trade_date = data.get("trade_date", "N/A")
    with cols[1]:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">%s</div>'
            '<div class="metric-label">分析日期</div></div>' % trade_date,
            unsafe_allow_html=True,
        )
    inv_debate = data.get("investment_debate_state", {})
    if not isinstance(inv_debate, dict):
        inv_debate = {}
    inv_decision = extract_decision(inv_debate.get("judge_decision", ""))
    inv_color = "#ef4444" if "SELL" in inv_decision else "#22c55e" if "BUY" in inv_decision else "#eab308"
    inv_zh = EN_ZH_MAP.get(inv_decision, inv_decision)
    with cols[2]:
        st.markdown(
            '<div class="metric-card"><div class="metric-value" style="color:%s">%s</div>'
            '<div class="metric-label">研究团队建议</div></div>' % (inv_color, inv_zh),
            unsafe_allow_html=True,
        )
    final = data.get("final_trade_decision", "")
    final_decision = extract_decision(final)
    final_color = "#ef4444" if "SELL" in final_decision else "#22c55e" if "BUY" in final_decision else "#eab308"
    final_zh = EN_ZH_MAP.get(final_decision, final_decision)
    with cols[3]:
        st.markdown(
            '<div class="metric-card"><div class="metric-value" style="color:%s">%s</div>'
            '<div class="metric-label">最终交易决策</div></div>' % (final_color, final_zh),
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Page: Historical Analysis View
# ─────────────────────────────────────────────
def render_history_page(data):
    """Render the full analysis page for historical data."""
    render_overview_metrics(data)
    st.markdown("---")

    tabs = st.tabs([
        "🏛️ 总览",
        "⚔️ 投资辩论",
        "🛡️ 风控辩论",
        "📊 分析师报告",
        "📋 原始数据",
    ])

    with tabs[0]:
        render_tab_overview(data)
    with tabs[1]:
        render_tab_investment_debate(data)
    with tabs[2]:
        render_tab_risk_debate(data)
    with tabs[3]:
        render_tab_analyst_reports(data)
    with tabs[4]:
        st.markdown("## 📋 原始 JSON 数据")
        st.json(data)


def render_tab_overview(data):
    st.markdown("## 📌 分析流水线总览")
    st.markdown("""
```
┌────────────┐   ┌────────────┐   ┌────────┐   ┌────────────┐   ┌────────────┐
│ I. 分析师   │-->│ II. 研究   │-->│III.交易 │-->│ IV. 风控   │-->│ V. 组合决策 │
│ 市场/社交/  │   │ 多空辩论   │   │  交易员 │   │ 三方辩论   │   │ 组合经理   │
│ 新闻/基本面 │   │ + 裁决     │   │        │   │ + 裁决     │   │            │
└────────────┘   └────────────┘   └────────┘   └────────────┘   └────────────┘
```
    """)

    inv_debate = data.get("investment_debate_state", {})
    if not isinstance(inv_debate, dict):
        inv_debate = {}
    risk_debate = data.get("risk_debate_state", {})
    if not isinstance(risk_debate, dict):
        risk_debate = {}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⚔️ 投资辩论结果")
        judge = inv_debate.get("judge_decision", "")
        if judge:
            decision = extract_decision(judge)
            st.markdown(get_decision_badge(decision), unsafe_allow_html=True)
            judge_display = judge[:800] + "..." if len(judge) > 800 else judge
            st.markdown(translate_to_chinese(judge_display))
            with st.expander("📝 查看英文原文", expanded=False):
                st.markdown(judge_display)
        else:
            st.info("无投资辩论数据")

    with col2:
        st.markdown("### 🛡️ 风控辩论结果")
        risk_judge = risk_debate.get("judge_decision", "")
        if risk_judge:
            decision = extract_decision(risk_judge)
            st.markdown(get_decision_badge(decision), unsafe_allow_html=True)
            risk_display = risk_judge[:800] + "..." if len(risk_judge) > 800 else risk_judge
            st.markdown(translate_to_chinese(risk_display))
            with st.expander("📝 查看英文原文", expanded=False):
                st.markdown(risk_display)
        else:
            st.info("无风控辩论数据")

    st.markdown("---")
    st.markdown("### 🎯 最终交易决策")
    final = data.get("final_trade_decision", "")
    if final:
        decision = extract_decision(final)
        st.markdown(get_decision_badge(decision), unsafe_allow_html=True)
        st.markdown(translate_to_chinese(final))
        with st.expander("📝 查看英文原文", expanded=False):
            st.markdown(final)
    else:
        st.info("无最终决策数据")


def render_tab_investment_debate(data):
    st.markdown("## ⚔️ 投资辩论（看多方 vs 看空方）")
    st.markdown("> 看多方与看空方展开激烈辩论，研究主管进行最终裁决")
    inv_debate = data.get("investment_debate_state", {})
    if not isinstance(inv_debate, dict):
        inv_debate = {}
    if not inv_debate:
        st.info("此次分析没有投资辩论数据")
        return

    bull = inv_debate.get("bull_history", "")
    if bull:
        render_debate_card("看多方 (Bull Analyst)", "🐂", bull, "bull-card")
    bear = inv_debate.get("bear_history", "")
    if bear:
        render_debate_card("看空方 (Bear Analyst)", "🐻", bear, "bear-card")
    judge = inv_debate.get("judge_decision", "")
    if judge:
        st.markdown("---")
        st.markdown("### ⚖️ 研究主管裁决 (Research Manager)")
        render_debate_card("研究主管裁决", "⚖️", judge, "judge-card")

    # Show full debate history if available
    history = inv_debate.get("history", "")
    if history:
        with st.expander("📜 完整辩论记录（中文翻译）", expanded=False):
            st.markdown(translate_to_chinese(history))
            with st.expander("📝 查看英文原文", expanded=False):
                st.markdown(history)


def render_tab_risk_debate(data):
    st.markdown("## 🛡️ 风控辩论（激进 vs 保守 vs 中立）")
    st.markdown("> 激进派、保守派、中立派三方辩论，组合经理进行最终裁决")
    risk_debate = data.get("risk_debate_state", {})
    if not isinstance(risk_debate, dict):
        risk_debate = {}
    if not risk_debate:
        st.info("此次分析没有风控辩论数据")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        agg = risk_debate.get("aggressive_history", "")
        if agg:
            render_debate_card("激进派 (Aggressive)", "🔥", agg, "aggressive-card")
    with col2:
        con = risk_debate.get("conservative_history", "")
        if con:
            render_debate_card("保守派 (Conservative)", "🛡️", con, "conservative-card")
    with col3:
        neu = risk_debate.get("neutral_history", "")
        if neu:
            render_debate_card("中立派 (Neutral)", "⚖️", neu, "neutral-card")

    risk_judge = risk_debate.get("judge_decision", "")
    if risk_judge:
        st.markdown("---")
        st.markdown("### 📊 组合经理最终裁决 (Portfolio Manager)")
        render_debate_card("组合经理最终裁决", "📊", risk_judge, "portfolio-card")

    history = risk_debate.get("history", "")
    if history:
        with st.expander("📜 完整辩论记录（中文翻译）", expanded=False):
            st.markdown(translate_to_chinese(history))
            with st.expander("📝 查看英文原文", expanded=False):
                st.markdown(history)


def render_tab_analyst_reports(data):
    st.markdown("## 📊 分析师报告")
    col1, col2 = st.columns(2)
    with col1:
        render_analyst_report("市场技术分析", "📈", data.get("market_report", ""))
        render_analyst_report("新闻分析", "📰", data.get("news_report", ""))
    with col2:
        render_analyst_report("社交情绪分析", "💬", data.get("sentiment_report", ""))
        render_analyst_report("基本面分析", "📊", data.get("fundamentals_report", ""))

    trader = data.get("trader_investment_decision", "") or data.get("trader_investment_plan", "")
    if trader:
        st.markdown("---")
        render_analyst_report("交易员决策", "💰", trader)


# ─────────────────────────────────────────────
# Page: Live Analysis
# ─────────────────────────────────────────────
def render_live_page():
    """Render the live analysis tracking page."""
    live_state = load_live_state()

    if live_state is None:
        st.markdown("""
## 🔴 实时分析

当前没有正在运行的分析任务。

### 如何启动实时分析

在终端中运行：

```bash
python live_analysis_runner.py --ticker NVDA --date 2025-03-20 --config deepseek
```

参数说明：
- `--ticker`: 股票代码（如 NVDA, AAPL, TSLA）
- `--date`: 分析日期（YYYY-MM-DD 格式）
- `--config`: LLM 配置（deepseek 或 openai）

启动后，此页面会自动刷新显示实时进展。
        """)

        # Auto-refresh button
        if st.button("🔄 刷新检测"):
            pass  # Streamlit will rerun

        return

    # ─── Live Analysis Display ───
    status = live_state.get("status", "unknown")
    ticker = live_state.get("ticker", "N/A")
    trade_date = live_state.get("trade_date", "N/A")
    updated_at = live_state.get("updated_at", "N/A")
    current_stage = live_state.get("current_stage", "starting")

    # Status banner
    if status == "running":
        st.markdown(
            '<div class="live-banner">'
            '<span class="live-dot"></span> '
            '<strong>实时分析进行中</strong> &mdash; '
            '%s | %s | 更新于 %s'
            '</div>' % (ticker, trade_date, updated_at),
            unsafe_allow_html=True,
        )
    elif status == "completed":
        st.success("✅ 分析已完成 — %s | %s | 完成于 %s" % (ticker, trade_date, updated_at))
    elif status == "error":
        st.error("❌ 分析出错 — %s" % live_state.get("error", "未知错误"))
    elif status == "initializing":
        st.info("⏳ 正在初始化分析... %s | %s" % (ticker, trade_date))

    # Pipeline progress
    st.markdown("### 📊 分析流水线进度")
    render_pipeline_progress(current_stage)

    chunk_count = live_state.get("chunk_count", 0)
    if chunk_count:
        st.markdown("*已处理 %d 个节点状态更新*" % chunk_count)

    st.markdown("---")

    # ─── Show available data based on current progress ───
    tab_names = ["📊 实时总览"]
    inv_ds = live_state.get("investment_debate_state", {})
    has_inv_debate = isinstance(inv_ds, dict) and (inv_ds.get("bull_history") or inv_ds.get("bear_history"))
    if has_inv_debate:
        tab_names.append("⚔️ 投资辩论")

    risk_ds = live_state.get("risk_debate_state", {})
    has_risk_debate = isinstance(risk_ds, dict) and (
        risk_ds.get("aggressive_history") or risk_ds.get("conservative_history")
    )
    if has_risk_debate:
        tab_names.append("🛡️ 风控辩论")

    has_reports = any(live_state.get(k) for k in ["market_report", "sentiment_report", "news_report", "fundamentals_report"])
    if has_reports:
        tab_names.append("📊 分析师报告")

    live_tabs = st.tabs(tab_names)
    tab_idx = 0

    with live_tabs[tab_idx]:
        render_live_overview(live_state)
    tab_idx += 1

    if has_inv_debate:
        with live_tabs[tab_idx]:
            render_tab_investment_debate(live_state)
        tab_idx += 1

    if has_risk_debate:
        with live_tabs[tab_idx]:
            render_tab_risk_debate(live_state)
        tab_idx += 1

    if has_reports:
        with live_tabs[tab_idx]:
            render_tab_analyst_reports(live_state)

    # Auto-refresh for running state
    if status in ("running", "initializing"):
        st.markdown("---")
        st.markdown("*⏱ 页面将每 5 秒自动刷新...*")
        time.sleep(5)
        st.rerun()


def render_live_overview(data):
    """Render live analysis overview."""
    st.markdown("## 📊 实时分析总览")

    # Show what we have so far
    stage_data = []

    # Analyst reports
    for key, name, icon in [
        ("market_report", "市场分析师报告", "📈"),
        ("sentiment_report", "社交情绪分析报告", "💬"),
        ("news_report", "新闻分析报告", "📰"),
        ("fundamentals_report", "基本面分析报告", "📊"),
    ]:
        content = data.get(key, "")
        if content:
            decision = extract_decision(content[-500:] if len(content) > 500 else content)
            badge = get_decision_badge(decision)
            st.markdown("%s **%s** %s" % (icon, name, badge), unsafe_allow_html=True)
            with st.expander("查看报告详情（中文翻译）", expanded=False):
                display_content = content[:2000]
                st.markdown(translate_to_chinese(display_content))
                with st.expander("📝 查看英文原文", expanded=False):
                    st.markdown(display_content)
            st.markdown("")

    # Investment debate status
    inv_ds = data.get("investment_debate_state", {})
    if isinstance(inv_ds, dict):
        bull = inv_ds.get("bull_history", "")
        bear = inv_ds.get("bear_history", "")
        judge = inv_ds.get("judge_decision", "")
        count = inv_ds.get("count", 0)
        if bull or bear:
            st.markdown("---")
            st.markdown("### ⚔️ 投资辩论 (第 %d 回合)" % count)
            col1, col2 = st.columns(2)
            with col1:
                if bull:
                    st.markdown("**🐂 看多方最新论述：**")
                    bull_text = bull[-800:] if len(bull) > 800 else bull
                    st.markdown(translate_to_chinese(bull_text))
                    with st.expander("📝 英文原文", expanded=False):
                        st.markdown(bull_text)
            with col2:
                if bear:
                    st.markdown("**🐻 看空方最新论述：**")
                    bear_text = bear[-800:] if len(bear) > 800 else bear
                    st.markdown(translate_to_chinese(bear_text))
                    with st.expander("📝 英文原文", expanded=False):
                        st.markdown(bear_text)
            if judge:
                st.markdown("**⚖️ 研究主管裁决：**")
                decision = extract_decision(judge)
                st.markdown(get_decision_badge(decision), unsafe_allow_html=True)
                st.markdown(translate_to_chinese(judge[:1000]))
                with st.expander("📝 查看英文原文", expanded=False):
                    st.markdown(judge[:1000])

    # Trader
    trader = data.get("trader_investment_plan", "")
    if trader:
        st.markdown("---")
        st.markdown("### 💰 交易员决策")
        st.markdown(translate_to_chinese(trader[:1000]))
        with st.expander("📝 查看英文原文", expanded=False):
            st.markdown(trader[:1000])

    # Risk debate status
    risk_ds = data.get("risk_debate_state", {})
    if isinstance(risk_ds, dict):
        agg = risk_ds.get("aggressive_history", "")
        con = risk_ds.get("conservative_history", "")
        neu = risk_ds.get("neutral_history", "")
        rj = risk_ds.get("judge_decision", "")
        count = risk_ds.get("count", 0)
        if agg or con or neu:
            st.markdown("---")
            st.markdown("### 🛡️ 风控辩论 (第 %d 回合)" % count)
            col1, col2, col3 = st.columns(3)
            with col1:
                if agg:
                    st.markdown("**🔥 激进派：**")
                    agg_text = agg[-600:] if len(agg) > 600 else agg
                    st.markdown(translate_to_chinese(agg_text))
                    with st.expander("📝 英文原文", expanded=False):
                        st.markdown(agg_text)
            with col2:
                if con:
                    st.markdown("**🛡️ 保守派：**")
                    con_text = con[-600:] if len(con) > 600 else con
                    st.markdown(translate_to_chinese(con_text))
                    with st.expander("📝 英文原文", expanded=False):
                        st.markdown(con_text)
            with col3:
                if neu:
                    st.markdown("**⚖️ 中立派：**")
                    neu_text = neu[-600:] if len(neu) > 600 else neu
                    st.markdown(translate_to_chinese(neu_text))
                    with st.expander("📝 英文原文", expanded=False):
                        st.markdown(neu_text)
            if rj:
                st.markdown("**📊 组合经理裁决：**")
                decision = extract_decision(rj)
                st.markdown(get_decision_badge(decision), unsafe_allow_html=True)
                st.markdown(translate_to_chinese(rj[:1000]))
                with st.expander("📝 查看英文原文", expanded=False):
                    st.markdown(rj[:1000])

    # Final decision
    final = data.get("final_trade_decision", "")
    if final:
        st.markdown("---")
        st.markdown("### 🎯 最终交易决策")
        decision = extract_decision(final)
        st.markdown(get_decision_badge(decision), unsafe_allow_html=True)
        st.markdown(translate_to_chinese(final))
        with st.expander("📝 查看英文原文", expanded=False):
            st.markdown(final)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar and return (mode, data, file_info, date)."""
    with st.sidebar:
        st.markdown("# 📊 TradingAgents")
        st.markdown("#### Web Dashboard")
        st.markdown("---")

        mode = st.radio(
            "选择模式",
            ["🔴 实时分析", "📂 查看历史记录", "📤 上传 JSON 文件"],
            index=0,
        )

        st.markdown("---")

        if mode == "🔴 实时分析":
            live_state = load_live_state()
            if live_state and live_state.get("status") in ("running", "initializing"):
                st.markdown("**⚡ 分析进行中**")
                st.markdown("股票: **%s**" % live_state.get("ticker", "N/A"))
                st.markdown("日期: **%s**" % live_state.get("trade_date", "N/A"))
                st.markdown("阶段: **%s**" % live_state.get("current_stage", "starting"))
            elif live_state and live_state.get("status") == "completed":
                st.markdown("**✅ 上次分析已完成**")
                st.markdown("股票: **%s**" % live_state.get("ticker", "N/A"))
            else:
                st.markdown("**💤 无运行中的分析**")

            st.markdown("---")
            st.markdown("**启动分析命令：**")
            st.code("python live_analysis_runner.py \\\n  --ticker NVDA \\\n  --date 2025-03-20 \\\n  --config deepseek")

            return "live", None, None, None

        elif mode == "📂 查看历史记录":
            log_files = find_log_files()
            if not log_files:
                st.warning("未找到历史记录。\n\n请先运行分析。")
                return "history", None, None, None

            tickers = sorted(set(f["ticker"] for f in log_files))
            selected_ticker = st.selectbox("🏷️ 选择股票", ["全部"] + tickers)
            filtered = log_files
            if selected_ticker != "全部":
                filtered = [f for f in log_files if f["ticker"] == selected_ticker]
            if not filtered:
                st.info("没有匹配的记录")
                return "history", None, None, None

            labels = [f["label"] for f in filtered]
            selected_label = st.selectbox("📅 选择分析记录", labels)
            selected_file = next(f for f in filtered if f["label"] == selected_label)

            all_data, dates = load_log_data(selected_file["path"])
            if not dates:
                st.error("无法解析该文件")
                return "history", None, None, None

            if len(dates) > 1:
                selected_date = st.selectbox("📆 选择日期", dates)
            else:
                selected_date = dates[0]

            st.markdown("---")
            st.markdown("**文件:** `%s`" % selected_file["path"])
            return "history", all_data.get(selected_date, {}), selected_file, selected_date

        else:  # Upload mode
            uploaded = st.file_uploader("上传 JSON 日志文件", type=["json"])
            if uploaded:
                try:
                    data = json.load(uploaded)
                    if isinstance(data, dict):
                        dates = list(data.keys())
                        if dates:
                            selected_date = st.selectbox("📆 选择日期", dates)
                            return "history", data.get(selected_date, {}), {"label": uploaded.name}, selected_date
                except (json.JSONDecodeError, ValueError):
                    st.error("无法解析上传的 JSON 文件")
            return "history", None, None, None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    mode, data, file_info, selected_date = render_sidebar()

    if mode == "live":
        render_live_page()
    elif data:
        ticker = data.get("company_of_interest", "N/A")
        trade_date = data.get("trade_date", selected_date or "N/A")
        st.markdown("# 📊 TradingAgents 分析报告")
        st.markdown("### %s — %s" % (ticker, trade_date))
        render_history_page(data)
    else:
        # Welcome page
        st.markdown("""
# 📊 TradingAgents Web Dashboard

### 欢迎使用 TradingAgents Web 仪表板！

在浏览器中查看完整的 **AI 多智能体辩论过程** 和 **分析报告**，所有内容附带中文注释。

---

#### 🔴 实时分析（新功能！）

在左侧选择「🔴 实时分析」模式，然后在终端启动分析：

```bash
python live_analysis_runner.py --ticker NVDA --date 2025-03-20 --config deepseek
```

仪表板会 **每 5 秒自动刷新**，实时展示：
- 📊 每个分析师的报告进展
- ⚔️ 投资辩论（看多 vs 看空）实时内容
- 🛡️ 风控辩论（激进/保守/中立）实时内容
- 🎯 最终决策

---

#### 📂 查看历史记录

在左侧选择「📂 查看历史记录」，浏览之前的分析结果。

---

#### 🌐 中文翻译

所有英文辩论内容和分析报告都会自动添加中文注释，包括：
- 金融术语翻译（BUY→买入, SELL→卖出 等）
- 标题翻译
- 关键词注释
        """)

        log_files = find_log_files()
        if log_files:
            st.markdown("---")
            st.markdown("### 📁 发现 %d 条历史记录" % len(log_files))
            for f in log_files[:10]:
                st.markdown("- **%s** — `%s`" % (f["label"], f["path"]))

        live_state = load_live_state()
        if live_state and live_state.get("status") in ("running", "initializing"):
            st.markdown("---")
            st.warning("⚡ 检测到正在运行的分析任务！请切换到「🔴 实时分析」模式查看。")


if __name__ == "__main__":
    main()
