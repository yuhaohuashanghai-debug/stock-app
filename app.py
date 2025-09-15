import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
import openai
import datetime

# ========== OpenAI Key配置 ==========
if "OPENAI_API_KEY" in st.secrets.get("OPENAI", {}):
    openai.api_key = st.secrets["OPENAI"]["OPENAI_API_KEY"]
else:
    openai.api_key = st.text_input("请输入你的OpenAI API KEY", type="password")

st.set_page_config(page_title="A股智能分析", layout="wide")
st.title("📈 A股实时智能技术分析 & AI趋势预测")

# ========== 输入区 ==========
stock_code = st.text_input("请输入A股股票代码（如 600519 或 000001）：", value="600519")
start_date = st.date_input("选择起始日期", value=datetime.date(2022,1,1))

# ========== 数据抓取 ==========
@st.cache_data(ttl=900)
def fetch_kline(code, start_date):
    code = code.strip()
    if len(code) != 6:
        return pd.DataFrame()
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date.strftime("%Y%m%d"), adjust="qfq")
        df.rename(columns={"日期":"date", "开盘":"open", "收盘":"close", "最高":"high", "最低":"low", "成交量":"volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        return df
    except Exception as e:
        st.warning("数据获取失败："+str(e))
        return pd.DataFrame()

df = fetch_kline(stock_code, start_date)
if df.empty:
    st.error("未获取到有效行情数据，请检查股票代码或起始日期！")
    st.stop()

# ========== 技术指标 ==========
df.ta.sma(length=5, append=True)
df.ta.sma(length=10, append=True)
df.ta.sma(length=20, append=True)
df.ta.macd(append=True)
df.ta.rsi(length=14, append=True)
df["sma5"] = df["SMA_5"]
df["sma10"] = df["SMA_10"]
df["sma20"] = df["SMA_20"]
df["macd"] = df["MACD_12_26_9"]
df["macds"] = df["MACDs_12_26_9"]
df["macdh"] = df["MACDh_12_26_9"]
df["rsi"] = df["RSI_14"]

# ========== K线+均线图 ==========
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K线"))
fig.add_trace(go.Scatter(x=df["date"], y=df["sma5"], mode="lines", name="MA5"))
fig.add_trace(go.Scatter(x=df["date"], y=df["sma10"], mode="lines", name="MA10"))
fig.add_trace(go.Scatter(x=df["date"], y=df["sma20"], mode="lines", name="MA20"))
fig.update_layout(height=450, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ========== MACD、RSI图 ==========
col1, col2 = st.columns(2)
with col1:
    st.subheader("MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(x=df["date"], y=df["macdh"], name="MACD柱"))
    fig_macd.add_trace(go.Scatter(x=df["date"], y=df["macd"], mode="lines", name="MACD线"))
    fig_macd.add_trace(go.Scatter(x=df["date"], y=df["macds"], mode="lines", name="信号线"))
    st.plotly_chart(fig_macd, use_container_width=True)

with col2:
    st.subheader("RSI")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["rsi"], mode="lines", name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(fig_rsi, use_container_width=True)

# ========== 展示最新技术指标 ==========
st.markdown("### 最近5日关键技术指标")
st.dataframe(df[["date","close","sma5","sma10","sma20","macd","macds","macdh","rsi"]].tail(5), use_container_width=True)

# ========== AI智能趋势预测 ==========
with st.expander("🔮 AI趋势分析（可选）", expanded=True):
    latest = df.tail(40) # 取近40日数据
    prompt = f"""你是专业A股分析师。请结合以下数据，进行短线技术面分析，并预测接下来5个交易日走势和操作建议：\n\n{latest[["date","close","sma5","sma10","sma20","macd","macds","macdh","rsi"]].to_string(index=False)}"""
    if st.button("生成AI趋势分析"):
        with st.spinner("AI智能分析中…"):
            try:
                completion = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role":"system","content":"你是专业A股技术分析专家。输出简明预测和建议，不用解释数据本身。"},
                              {"role":"user","content": prompt}],
                    temperature=0.2,
                    max_tokens=400
                )
                analysis = completion.choices[0].message.content
                st.markdown(f"**AI趋势预测结果：**\n\n{analysis}")
            except Exception as e:
                st.error("AI分析失败："+str(e))

st.info("本工具仅作技术演示，不构成投资建议。")
