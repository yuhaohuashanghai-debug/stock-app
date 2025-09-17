import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# ========== 页面配置 ==========
st.set_page_config(page_title="📈 实时股票AI分析平台", layout="wide")
st.title("📊 实时股票分析 + 资金流向 + 大盘板块对比 + AI 概率预测")

# ========== API Key 输入 ==========
DEEPSEEK_API_KEY = st.text_input("请输入 DeepSeek API Key（留空则只做本地技术点评）", type="password")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ========== 数据获取 ==========
@st.cache_data(ttl=300)
def fetch_realtime_kline(code: str):
    if code.startswith("6"):
        symbol = f"sh{code}"
    else:
        symbol = f"sz{code}"
    df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df.rename(columns={"date":"date","open":"open","close":"close",
                       "high":"high","low":"low","volume":"volume"}, inplace=True)
    return df

@st.cache_data(ttl=300)
def fetch_intraday_kline(code: str, period="60"):
    df = ak.stock_zh_a_hist(symbol=code, period=period, start_date="20240101", adjust="qfq")
    df.rename(columns={"日期":"date","开盘":"open","收盘":"close",
                       "最高":"high","最低":"low","成交量":"volume"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(120)

@st.cache_data(ttl=300)
def fetch_stock_news(code: str):
    try:
        df = ak.stock_news_em(symbol=code)
        for col in ["title", "新闻标题", "标题"]:
            if col in df.columns:
                return df[col].head(5).tolist()
        return ["未找到新闻标题字段"]
    except Exception as e:
        return [f"新闻获取失败: {e}"]

@st.cache_data(ttl=300)
def fetch_fund_flow(code: str):
    try:
        df = ak.stock_individual_fund_flow(stock=code)
        return df.tail(5)[["日期","主力净流入"]].to_dict("records")
    except Exception as e:
        return [{"error": str(e)}]

@st.cache_data(ttl=300)
def fetch_index_data(index_code="000300"):  # 默认沪深300
    df = ak.index_zh_a_hist(symbol=index_code, period="daily", start_date="20240101", adjust="qfq")
    df.rename(columns={"日期":"date","开盘":"open","收盘":"close",
                       "最高":"high","最低":"low","成交量":"volume"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600)
def get_stock_board(code: str):
    boards = ak.stock_board_concept_name_em()
    for b in boards["板块名称"].tolist():
        try:
            cons = ak.stock_board_concept_cons_em(symbol=b)
            if code in cons["代码"].tolist():
                return b
        except:
            continue
    return None

@st.cache_data(ttl=300)
def fetch_board_data(board_name: str):
    df = ak.stock_board_concept_hist_em(symbol=board_name, period="daily", start_date="20240101")
    df.rename(columns={"日期":"date","开盘":"open","收盘":"close",
                       "最高":"high","最低":"low","成交量":"volume"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ========== 技术指标 ==========
def add_indicators(df: pd.DataFrame, indicator: str):
    df["MA5"] = ta.sma(df["close"], length=5)
    df["MA20"] = ta.sma(df["close"], length=20)

    if indicator == "MACD":
        macd = ta.macd(df["close"])
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACDh"] = macd["MACDh_12_26_9"]
        df["MACDs"] = macd["MACDs_12_26_9"]
    elif indicator == "RSI":
        df["RSI"] = ta.rsi(df["close"], length=14)
    elif indicator == "BOLL":
        boll = ta.bbands(df["close"], length=20, std=2)
        df["BOLL_U"] = boll["BBU_20_2.0"]
        df["BOLL_M"] = boll["BBM_20_2.0"]
        df["BOLL_L"] = boll["BBL_20_2.0"]
    elif indicator == "KDJ":
        kdj = ta.stoch(df["high"], df["low"], df["close"])
        df["K"] = kdj["STOCHk_14_3_3"]
        df["D"] = kdj["STOCHd_14_3_3"]
        df["J"] = 3 * df["K"] - 2 * df["D"]

    return df.dropna()

# ========== 图表 ==========
def plot_chart(df: pd.DataFrame, code: str, indicator: str, show_ma: list, show_volume: bool):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05,
                        subplot_titles=(f"{code} K线及指标", indicator))

    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="K线图"), row=1, col=1)
    if "MA5" in show_ma:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], name="MA5"), row=1, col=1)
    if "MA20" in show_ma:
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], name="MA20"), row=1, col=1)
    if show_volume:
        fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量", opacity=0.4), row=1, col=1)

    if indicator == "MACD":
        fig.add_trace(go.Bar(x=df["date"], y=df["MACDh"], name="MACDh", opacity=0.3), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACDs"], name="信号线"), row=2, col=1)
    elif indicator == "RSI":
        fig.add_trace(go.Scatter(x=df["date"], y=df["RSI"], name="RSI"), row=2, col=1)
    elif indicator == "BOLL":
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_U"], name="上轨"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_M"], name="中轨"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_L"], name="下轨"), row=1, col=1)
    elif indicator == "KDJ":
        fig.add_trace(go.Scatter(x=df["date"], y=df["K"], name="K"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["D"], name="D"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["J"], name="J"), row=2, col=1)

    fig.update_layout(height=900, xaxis_rangeslider_visible=False, showlegend=True)
    return fig

# ========== 相对强弱 ==========
def compare_performance(stock_df, index_df, board_df):
    def pct(df): return (df["close"].iloc[-1] - df["close"].iloc[-6]) / df["close"].iloc[-6] * 100
    stock_pct = pct(stock_df)
    index_pct = pct(index_df)
    board_pct = pct(board_df) if board_df is not None else None
    return stock_pct, index_pct, board_pct

# ========== AI 概率预测 ==========
def deepseek_probability_predict(tech_summary, fund_flow, news_list, perf_compare, api_key):
    stock_pct, index_pct, board_pct = perf_compare
    perf_text = f"""
个股近5日涨幅: {stock_pct:.2f}%
沪深300近5日涨幅: {index_pct:.2f}%
板块近5日涨幅: {board_pct:.2f}%""" if board_pct is not None else f"""
个股近5日涨幅: {stock_pct:.2f}%
沪深300近5日涨幅: {index_pct:.2f}%
板块数据: 暂无"""

    flow_text = "\n".join([f"{d['日期']} 主力净流入: {d['主力净流入']}" for d in fund_flow if "主力净流入" in d])
    news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "无相关新闻"

    prompt = f"""
以下是某只股票的多维度数据，请结合日线+60分钟K线、资金流向、技术指标、新闻、大盘与板块对比，给出未来3日内的趋势概率预测：
- 上涨概率（%）
- 震荡概率（%）
- 下跌概率（%）
并简要说明原因。

【技术面】  
{tech_summary}

【资金流向】  
{flow_text if flow_text else "暂无资金流数据"}

【消息面】  
{news_text}

【相对强弱】  
{perf_text}
"""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat","messages": [{"role": "user", "content": prompt}],
               "max_tokens": 600,"temperature": 0.5}
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"DeepSeek 概率预测出错: {e}"

# ========== 主程序 ==========
code = st.text_input("请输入股票代码（如 600519）", "600519")
show_ma = st.multiselect("显示均线", ["MA5", "MA20"], default=["MA5", "MA20"])
show_volume = st.checkbox("显示成交量", value=True)
indicator = st.selectbox("选择额外指标", ["MACD", "RSI", "BOLL", "KDJ"])

if st.button("分析"):
    df = fetch_realtime_kline(code)
    df = add_indicators(df, indicator)

    st.plotly_chart(plot_chart(df, code, indicator, show_ma, show_volume), use_container_width=True)

    latest = df.iloc[-1]
    summary = f"收盘价:{latest['close']:.2f}, MA5:{latest['MA5']:.2f}, MA20:{latest['MA20']:.2f}"
    if indicator == "MACD":
        summary += f", MACD:{latest['MACD']:.3f}, 信号线:{latest['MACDs']:.3f}"
    st.subheader("📌 技术指标总结")
    st.write(summary)

    news_list = fetch_stock_news(code)
    st.subheader("📰 实时消息面")
    for n in news_list:
        st.write("- " + n)

    fund_flow = fetch_fund_flow(code)
    st.subheader("💰 资金流向（近5日）")
    for f in fund_flow:
        if "主力净流入" in f:
            st.write(f"{f['日期']} 主力净流入: {f['主力净流入']}")

    # 大盘 & 板块对比
    index_df = fetch_index_data("000300")  # 沪深300
    board_name = get_stock_board(code)
    board_df = fetch_board_data(board_name) if board_name else None
    stock_pct, index_pct, board_pct = compare_performance(df, index_df, board_df)

    st.subheader("📈 个股 vs 大盘 vs 板块")
    st.write(f"近5日个股涨幅: {stock_pct:.2f}%")
    st.write(f"近5日沪深300涨幅: {index_pct:.2f}%")
    if board_df is not None:
        st.write(f"近5日{board_name}板块涨幅: {board_pct:.2f}%")
    else:
        st.write("未找到所属板块数据")

    # AI 综合预测
    if DEEPSEEK_API_KEY:
        with st.spinner("DeepSeek AI 概率预测中..."):
            ai_text = deepseek_probability_predict(summary, fund_flow, news_list, (stock_pct, index_pct, board_pct), DEEPSEEK_API_KEY)
            st.subheader("📊 AI 趋势概率预测")
            st.write(ai_text)
    else:
        st.subheader("🤖 本地技术面点评")
        if indicator == "MACD":
            if latest["MACD"] > latest["MACDs"]:
                st.write("MACD 金叉，短期有反弹可能。")
            elif latest["MACD"] < latest["MACDs"]:
                st.write("MACD 死叉，短期下行动能较大。")
            else:
                st.write("MACD 持平，市场观望情绪浓。")
        elif indicator == "RSI":
            if latest["RSI"] < 30:
                st.write("RSI < 30，超卖区域，可能反弹。")
            elif latest["RSI"] > 70:
                st.write("RSI > 70，超买风险，可能回调。")
            else:
                st.write("RSI 中性，市场震荡。")
