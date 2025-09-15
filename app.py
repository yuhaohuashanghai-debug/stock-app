# app.py

import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="A股批量分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量智能技术分析 & AI趋势预测")

# --- 用户输入部分 ---
openai_key = st.text_input("请输入你的OpenAI API KEY（用于AI点评/趋势预测）", type="password")
codes_input = st.text_input("请输入A股股票代码（支持批量,如 600519,000977,588170）：", value="000977,518880")
start_date = st.date_input("选择起始日期", value=datetime.now().replace(year=2025, month=9, day=4))
ai_enable = st.toggle("启用AI趋势点评", value=True)
trend_days = st.selectbox("AI预测未来天数", options=[1, 3, 5, 7], index=1)

# --- AkShare数据自动适配股票/ETF ---
def fetch_ak_data(code, start_date):
    import akshare as ak
    import pandas as pd
    df = pd.DataFrame()
    try:
        # 先尝试A股股票接口
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date.strftime("%Y%m%d"), adjust="qfq")
        if not df.empty:
            df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                               "最高": "high", "最低": "low", "成交量": "volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
    except Exception:
        pass
    # 尝试新浪ETF接口（适配大部分ETF）
    try:
        df = ak.fund_etf_hist_sina(symbol=code)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= pd.to_datetime(start_date)]
            df = df.sort_values("date")
            df = df.reset_index(drop=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()

# --- 指标计算 ---
def calc_indicators(df):
    import pandas_ta as ta
    if "close" not in df.columns or len(df) < 20:
        return df
    try:
        df["SMA_5"] = ta.sma(df["close"], length=5)
        df["SMA_10"] = ta.sma(df["close"], length=10)
        df["SMA_20"] = ta.sma(df["close"], length=20)
        macd = ta.macd(df["close"])
        if macd is not None and not macd.empty:
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACDs"] = macd["MACDs_12_26_9"]
            df["MACDh"] = macd["MACDh_12_26_9"]
        df["RSI_6"] = ta.rsi(df["close"], length=6)
        df["RSI_12"] = ta.rsi(df["close"], length=12)
    except Exception as e:
        pass
    return df

# --- 图表展示 ---
def plot_kline(df, code):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="K线"))
    if "SMA_5" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_5"], mode='lines', name="SMA5"))
    if "SMA_10" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_10"], mode='lines', name="SMA10"))
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA_20"], mode='lines', name="SMA20"))
    fig.update_layout(title=f"{code} K线与均线", xaxis_rangeslider_visible=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    # MACD
    if "MACD" in df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df["date"], y=df["MACDh"], name="MACD柱"))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD线"))
        fig2.add_trace(go.Scatter(x=df["date"], y=df["MACDs"], name="信号线"))
        fig2.update_layout(title="MACD指标", height=200)
        st.plotly_chart(fig2, use_container_width=True)
    # RSI
    if "RSI_6" in df.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df["date"], y=df["RSI_6"], name="RSI6"))
        fig3.add_trace(go.Scatter(x=df["date"], y=df["RSI_12"], name="RSI12"))
        fig3.update_layout(title="RSI指标", height=200, yaxis=dict(range=[0,100]))
        st.plotly_chart(fig3, use_container_width=True)

# --- AI趋势预测 ---
def ai_trend_report(df, code, trend_days, openai_key):
    if not openai_key:
        return "未填写OpenAI KEY，无法生成AI趋势预测。"
    use_df = df.tail(60)[["date", "open", "close", "high", "low", "volume"]]
    data_str = use_df.to_csv(index=False)
    prompt = f"""
你是一位A股专业量化分析师。以下是{code}最近60日的每日行情（日期,开盘,收盘,最高,最低,成交量），请根据技术走势、成交量变化，预测该股未来{trend_days}日的涨跌趋势，并判断是否存在启动信号、买卖机会，请以精炼中文输出一份点评。数据如下（csv格式）：
{data_str}
"""
    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一位专业A股分析师。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.6,
        )
        return chat_completion.choices[0].message.content
    except Exception as ex:
        return f"AI分析调用失败：{ex}"

# --- 多策略信号批量选股模块 ---

def stock_signals(df):
    """为单只股票/ETF计算所有选股信号，返回信号标注和理由"""
    signals = []
    latest = df.iloc[-1]
    pre = df.iloc[-2] if len(df) >= 2 else latest

    # 1. 均线金叉
    if "SMA_5" in df.columns and "SMA_10" in df.columns:
        if pre["SMA_5"] < pre["SMA_10"] and latest["SMA_5"] > latest["SMA_10"]:
            signals.append("5日均线金叉10日均线")

    # 2. MACD金叉
    if "MACD" in df.columns and "MACDs" in df.columns:
        if pre["MACD"] < pre["MACDs"] and latest["MACD"] > latest["MACDs"]:
            signals.append("MACD金叉")

    # 3. RSI超卖反弹
    if "RSI_6" in df.columns and latest["RSI_6"] < 30 and pre["RSI_6"] >= 30:
        signals.append("RSI6超卖反弹")

    # 4. 放量突破（成交量比前5日均量大50%+今日涨幅>2%）
    if "volume" in df.columns and "close" in df.columns:
        if len(df) >= 6:
            pre_vol = df["volume"].iloc[-6:-1].mean()
            vol_up = latest["volume"] > 1.5 * pre_vol
            pct_chg = (latest["close"] - pre["close"]) / pre["close"] * 100 if pre["close"] > 0 else 0
            if vol_up and pct_chg > 2:
                signals.append("放量突破")
    
    # 5. 20日新高
    if "close" in df.columns and len(df) >= 20:
        if latest["close"] >= df["close"].iloc[-20:].max():
            signals.append("20日新高")

    # 6. 20日新低
    if "close" in df.columns and len(df) >= 20:
        if latest["close"] <= df["close"].iloc[-20:].min():
            signals.append("20日新低")

    return signals

# --- 主流程 ---
if st.button("批量分析"):
    codes = [c.strip() for c in codes_input.split(",") if c.strip()]
    for code in codes:
        st.header(f"【{code}】分析")
        df = fetch_ak_data(code, start_date)
        if df.empty:
            st.warning(f"{code} 数据未获取到，可能代码错误或日期过近。")
            continue
        df = calc_indicators(df)
        st.dataframe(df.tail(10))
        plot_kline(df, code)
        if ai_enable:
            with st.spinner(f"AI分析{code}中..."):
                ai_report = ai_trend_report(df, code, trend_days, openai_key)
                st.info(ai_report)
        st.divider()
else:
    st.markdown("> 支持多只A股代码批量技术分析+AI自动点评（如需AI预测请填写OpenAI KEY）")
