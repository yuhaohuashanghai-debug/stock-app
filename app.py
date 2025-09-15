# app.py

import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime
import openai

st.set_page_config(page_title="A股批量分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量智能技术分析 & AI趋势预测")

# --- 用户输入部分 ---
openai_key = st.text_input("请输入你的OpenAI API KEY（用于AI点评/趋势预测）", type="password")
codes_input = st.text_input("请输入A股股票代码（支持批量,如 600519,000977,588170）：", value="000977,518880")
start_date = st.date_input("选择起始日期", value=datetime.now().replace(year=2025, month=9, day=4))
ai_enable = st.toggle("启用AI趋势点评", value=True)
trend_days = st.selectbox("AI预测未来天数", options=[1, 3, 5, 7], index=1)

# --- AkShare获取行情数据 ---

def fetch_ak_data(code, start_date):
    import akshare as ak
    import pandas as pd
    df = pd.DataFrame()
    try:
        # 先尝试A股股票
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
