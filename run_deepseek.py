from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config for DeepSeek
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "deepseek"
config["deep_think_llm"] = "deepseek-chat"       # DeepSeek-V3
config["quick_think_llm"] = "deepseek-chat"       # DeepSeek-V3

config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1

# Use yfinance for all data (free, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# Analyze NVDA on a recent date
_, decision = ta.propagate("NVDA", "2025-03-20")
print("\n" + "=" * 60)
print("FINAL DECISION:", decision)
print("=" * 60)
