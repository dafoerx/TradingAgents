#!/usr/bin/env python3
"""
TradingAgents Live Analysis Runner
====================================
独立进程运行分析，实时将每个 Agent 节点的中间状态写入 JSON 文件，
供 Web 仪表板轮询读取并展示实时辩论进展。

用法:
    python live_analysis_runner.py --ticker NVDA --date 2025-03-20 [--config deepseek]
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────
# Live State File Management
# ─────────────────────────────────────────────
LIVE_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".live_state")


def get_live_state_path():
    """Return path to the live state JSON file."""
    return os.path.join(LIVE_STATE_DIR, "current_analysis.json")


def write_live_state(state_data):
    """Write current analysis state to the live state file (atomic)."""
    os.makedirs(LIVE_STATE_DIR, exist_ok=True)
    filepath = get_live_state_path()
    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2, default=str)
        # Atomic rename
        os.rename(tmp_path, filepath)
    except Exception as e:
        print("[WARN] Failed to write live state: {}".format(e))
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def clear_live_state():
    """Remove the live state file."""
    filepath = get_live_state_path()
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass


# ─────────────────────────────────────────────
# Pipeline Stage Detection
# ─────────────────────────────────────────────

# Ordered pipeline stages
PIPELINE_STAGES = [
    {"key": "market_report", "name_en": "Market Analyst", "name_zh": "市场分析师", "icon": "📈"},
    {"key": "sentiment_report", "name_en": "Social Media Analyst", "name_zh": "社交媒体分析师", "icon": "💬"},
    {"key": "news_report", "name_en": "News Analyst", "name_zh": "新闻分析师", "icon": "📰"},
    {"key": "fundamentals_report", "name_en": "Fundamentals Analyst", "name_zh": "基本面分析师", "icon": "📊"},
    {"key": "investment_debate", "name_en": "Investment Debate", "name_zh": "投资辩论", "icon": "⚔️"},
    {"key": "trader", "name_en": "Trader", "name_zh": "交易员", "icon": "💰"},
    {"key": "risk_debate", "name_en": "Risk Debate", "name_zh": "风控辩论", "icon": "🛡️"},
    {"key": "final_decision", "name_en": "Final Decision", "name_zh": "最终决策", "icon": "🎯"},
]


def detect_stage_from_state(state):
    """Detect the current pipeline stage from the state snapshot."""
    # Check from end to beginning (latest stage first)
    if state.get("final_trade_decision", ""):
        return "final_decision"
    risk_ds = state.get("risk_debate_state", {})
    if isinstance(risk_ds, dict) and (
        risk_ds.get("aggressive_history", "")
        or risk_ds.get("conservative_history", "")
        or risk_ds.get("neutral_history", "")
        or risk_ds.get("judge_decision", "")
    ):
        return "risk_debate"
    if state.get("trader_investment_plan", ""):
        return "trader"
    inv_ds = state.get("investment_debate_state", {})
    if isinstance(inv_ds, dict) and (
        inv_ds.get("bull_history", "")
        or inv_ds.get("bear_history", "")
        or inv_ds.get("judge_decision", "")
    ):
        return "investment_debate"
    if state.get("fundamentals_report", ""):
        return "fundamentals_report"
    if state.get("news_report", ""):
        return "news_report"
    if state.get("sentiment_report", ""):
        return "sentiment_report"
    if state.get("market_report", ""):
        return "market_report"
    return "starting"


def build_live_snapshot(state, ticker, trade_date, status="running", error_msg=""):
    """Build a snapshot dict from the current LangGraph state for the live state file."""
    inv_ds = state.get("investment_debate_state", {})
    if not isinstance(inv_ds, dict):
        inv_ds = {}
    risk_ds = state.get("risk_debate_state", {})
    if not isinstance(risk_ds, dict):
        risk_ds = {}

    current_stage = detect_stage_from_state(state)

    return {
        "status": status,  # "running", "completed", "error"
        "error": error_msg,
        "ticker": ticker,
        "trade_date": str(trade_date),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_stage": current_stage,
        # Analyst reports
        "market_report": state.get("market_report", ""),
        "sentiment_report": state.get("sentiment_report", ""),
        "news_report": state.get("news_report", ""),
        "fundamentals_report": state.get("fundamentals_report", ""),
        # Investment debate
        "investment_debate_state": {
            "bull_history": inv_ds.get("bull_history", ""),
            "bear_history": inv_ds.get("bear_history", ""),
            "history": inv_ds.get("history", ""),
            "current_response": inv_ds.get("current_response", ""),
            "judge_decision": inv_ds.get("judge_decision", ""),
            "count": inv_ds.get("count", 0),
        },
        # Trader
        "trader_investment_plan": state.get("trader_investment_plan", ""),
        # Risk debate
        "risk_debate_state": {
            "aggressive_history": risk_ds.get("aggressive_history", ""),
            "conservative_history": risk_ds.get("conservative_history", ""),
            "neutral_history": risk_ds.get("neutral_history", ""),
            "history": risk_ds.get("history", ""),
            "judge_decision": risk_ds.get("judge_decision", ""),
            "count": risk_ds.get("count", 0),
        },
        # Final
        "final_trade_decision": state.get("final_trade_decision", ""),
        "investment_plan": state.get("investment_plan", ""),
    }


# ─────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────

def run_analysis(ticker, trade_date, config_preset="deepseek"):
    """Run the full analysis pipeline with live state updates."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Build config
    config = DEFAULT_CONFIG.copy()
    if config_preset == "deepseek":
        config["llm_provider"] = "deepseek"
        config["deep_think_llm"] = "deepseek-chat"
        config["quick_think_llm"] = "deepseek-chat"
    elif config_preset == "openai":
        config["llm_provider"] = "openai"
        config["deep_think_llm"] = "gpt-4o"
        config["quick_think_llm"] = "gpt-4o-mini"

    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    print("=" * 60)
    print("[Live Runner] Ticker: {} | Date: {} | Config: {}".format(ticker, trade_date, config_preset))
    print("=" * 60)

    # Write initial state
    write_live_state({
        "status": "initializing",
        "ticker": ticker,
        "trade_date": str(trade_date),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_stage": "starting",
        "error": "",
    })

    try:
        # Create graph (always debug=True to enable streaming)
        ta = TradingAgentsGraph(debug=True, config=config)

        # Create initial state
        init_agent_state = ta.propagator.create_initial_state(ticker, trade_date)
        args = ta.propagator.get_graph_args()

        print("[Live Runner] Starting graph.stream()...")
        write_live_state({
            "status": "running",
            "ticker": ticker,
            "trade_date": str(trade_date),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_stage": "starting",
            "error": "",
        })

        # Stream through the graph, writing live state after each chunk
        final_state = None
        chunk_count = 0
        for chunk in ta.graph.stream(init_agent_state, **args):
            chunk_count += 1
            final_state = chunk

            # Build and write live snapshot
            snapshot = build_live_snapshot(chunk, ticker, trade_date, status="running")
            snapshot["chunk_count"] = chunk_count
            write_live_state(snapshot)

            # Print progress
            stage = snapshot["current_stage"]
            stage_info = next((s for s in PIPELINE_STAGES if s["key"] == stage), None)
            if stage_info:
                print("[Chunk {}] Stage: {} {}".format(
                    chunk_count, stage_info["icon"], stage_info["name_zh"]
                ))

        if final_state is None:
            raise RuntimeError("Graph stream produced no output")

        # Write final completed state
        snapshot = build_live_snapshot(final_state, ticker, trade_date, status="completed")
        snapshot["chunk_count"] = chunk_count
        write_live_state(snapshot)

        # Also log state via the standard mechanism
        ta.curr_state = final_state
        ta._log_state(trade_date, final_state)

        print("\n" + "=" * 60)
        decision = ta.process_signal(final_state.get("final_trade_decision", ""))
        print("[Live Runner] COMPLETED! Decision: {}".format(decision))
        print("=" * 60)
        return decision

    except Exception as e:
        error_msg = traceback.format_exc()
        print("[Live Runner] ERROR: {}".format(e))
        print(error_msg)
        write_live_state({
            "status": "error",
            "ticker": ticker,
            "trade_date": str(trade_date),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_stage": "error",
            "error": str(e),
        })
        return None


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingAgents Live Analysis Runner")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Stock ticker symbol")
    parser.add_argument("--date", type=str, default="2025-03-20", help="Trade date (YYYY-MM-DD)")
    parser.add_argument("--config", type=str, default="deepseek",
                        choices=["deepseek", "openai"],
                        help="LLM config preset")
    args = parser.parse_args()

    run_analysis(args.ticker, args.date, args.config)
