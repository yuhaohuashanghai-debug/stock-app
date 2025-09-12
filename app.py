import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai

# ✅ 设置 OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="📊 A股策略分析", layout="wide")
st.title("📈 基于 AkShare + ChatGPT 的 A股技术分析与趋势预测")


# --- 获取行情数据 ---
@st.cache_data(ttl=3600)
def fetch_data(code: str, start_date="20220101"):
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, adjust="qfq")
        df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume"
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        return df
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return pd.DataFrame()


# --- 技术指标计算 ---
def add_indicators(df):
    df["MA5"] = ta.sma(df["close"], length=5)
    df["MA10"] = ta.sma(df["close"], length=10)
    df["MA20"] = ta.sma(df["close"], length=20)

    macd = ta.macd(df["close"])
    if macd is not None and not macd.empty:
        df["MACD"] = macd.get("MACD_12_26_9", None)
        df["MACD_H"] = macd.get("MACDh_12_26_9", None)
        df["MACD_S"] = macd.get("MACDs_12_26_9", None)
    else:
        df["MACD"], df["MACD_H"], df["MACD_S"] = None, None, None

    rsi = ta.rsi(df["close"], length=14)
    df["RSI"] = rsi if rsi is not None and not rsi.empty else None

    boll = ta.bbands(df["close"], length=20, std=2)
    if boll is not None and not boll.empty:
        up_col = next((c for c in boll.columns if "BBU" in c), None)
        mid_col = next((c for c in boll.columns if "BBM" in c), None)
        low_col = next((c for c in boll.columns if "BBL" in c), None)
        df["BOLL_UP"] = boll[up_col] if up_col else None
        df["BOLL_MID"] = boll[mid_col] if mid_col else None
        df["BOLL_LOW"] = boll[low_col] if low_col else None
    else:
        df["BOLL_UP"], df["BOLL_MID"], df["BOLL_LOW"] = None, None, None

    return df


# --- 趋势预测 ---
def predict_trend(df):
    latest = df.iloc[-1]
    signals = []

    # MACD
    try:
        if pd.notna(latest["MACD"]) and pd.notna(latest["MACD_S"]):
            if latest["MACD"] > latest["MACD_S"]:
                signals.append("MACD 金叉 → 看涨")
            else:
                signals.append("MACD 死叉 → 看跌")
        else:
            signals.append("⚠️ MACD 数据不足，无法判断")
    except Exception:
        signals.append("⚠️ MACD 计算失败")

    # RSI
    try:
        if pd.notna(latest["RSI"]):
            if latest["RSI"] < 30:
                signals.append("RSI < 30 → 超卖反弹概率大")
            elif latest["RSI"] > 70:
                signals.append("RSI > 70 → 超买回落概率大")
            else:
                signals.append("RSI 在正常区间 → 市场相对平稳")
        else:
            signals.append("⚠️ RSI 数据不足，无法判断")
    except Exception:
        signals.append("⚠️ RSI 计算失败")

    # BOLL
    try:
        if pd.notna(latest["BOLL_UP"]) and pd.notna(latest["BOLL_LOW"]):
            if latest["close"] > latest["BOLL_UP"]:
                signals.append("股价突破布林上轨 → 短期或回调")
            elif latest["close"] < latest["BOLL_LOW"]:
                signals.append("股价跌破布林下轨 → 可能反弹")
            else:
                signals.append("股价位于布林带中轨附近 → 区间震荡")
        else:
            signals.append("⚠️ BOLL 数据不足，无法判断")
    except Exception:
        signals.append("⚠️ BOLL 计算失败")

    return signals


# --- 策略回测 ---
def backtest_macd(df, lookback=90, holding_days=5):
    results = {"金叉": {"次数": 0, "胜率": 0}, "死叉": {"次数": 0, "胜率": 0}}
    trades = []

    if "MACD" not in df.columns or "MACD_S" not in df.columns:
        return results, trades

    df = df.dropna().reset_index(drop=True)
    df = df.iloc[-lookback:]

    for i in range(1, len(df) - holding_days):
        today = df.iloc[i]
        yesterday = df.iloc[i - 1]

        # 金叉
        if yesterday["MACD"] <= yesterday["MACD_S"] and today["MACD"] > today["MACD_S"]:
            entry_price = today["close"]
            exit_price = df.iloc[i + holding_days]["close"]
            ret = (exit_price - entry_price) / entry_price
            trades.append(("金叉", today["date"], entry_price, exit_price, ret))
            results["金叉"]["次数"] += 1
            if ret > 0:
                results["金叉"]["胜率"] += 1

        # 死叉
        if yesterday["MACD"] >= yesterday["MACD_S"] and today["MACD"] < today["MACD_S"]:
            entry_price = today["close"]
            exit_price = df.iloc[i + holding_days]["close"]
            ret = (exit_price - entry_price) / entry_price
            trades.append(("死叉", today["date"], entry_price, exit_price, ret))
            results["死叉"]["次数"] += 1
            if ret < 0:
                results["死叉"]["胜率"] += 1

    if results["金叉"]["次数"] > 0:
        results["金叉"]["胜率"] = results["金叉"]["胜率"] / results["金叉"]["次数"]
    if results["死叉"]["次数"] > 0:
        results["死叉"]["胜率"] = results["死叉"]["胜率"] / results["死叉"]["次数"]

    return results, trades


# --- ChatGPT 投资解读 ---
def ai_analysis(code, df, signals):
    latest = df.iloc[-1]
    prompt = f"""
你是一名专业的A股分析师，请根据以下数据写一份简短的研报风格解读，内容包含：技术面分析、风险提示、未来一周走势判断。
股票代码: {code}
日期: {latest['date'].strftime('%Y-%m-%d')}
收盘价: {latest['close']}
MA5: {latest['MA5']:.2f}, MA10: {latest['MA10']:.2f}, MA20: {latest['MA20']:.2f}
MACD: {latest['MACD']}, Signal: {latest['MACD_S']}
RSI: {latest['RSI']}
BOLL: 上轨 {latest['BOLL_UP']}, 中轨 {latest['BOLL_MID']}, 下轨 {latest['BOLL_LOW']}
信号总结: {"; ".join(signals)}
要求：语言专业、简洁，面向投资者，不要超过300字。
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "你是专业的证券分析师。"},
                      {"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.6
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ ChatGPT 分析失败: {e}"


# --- 页面交互 ---
code = st.text_input("请输入6位股票代码", value="000001")

if st.button("分析股票"):
    df = fetch_data(code)
    if not df.empty:
        df = add_indicators(df)

        # 绘制图表
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25])
        fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"],
                                     low=df["low"], close=df["close"], name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA5"], name="MA5"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA10"], name="MA10"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], name="MA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_UP"], name="BOLL_UP", line=dict(dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_MID"], name="BOLL_MID", line=dict(dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["BOLL_LOW"], name="BOLL_LOW", line=dict(dash="dot")), row=1, col=1)

        fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="成交量"), row=2, col=1)
        fig.add_trace(go.Bar(x=df["date"], y=df["MACD_H"], name="MACD柱状"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["MACD_S"], name="信号线"), row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # 趋势预测
        st.subheader("📌 技术信号解读")
        signals = predict_trend(df)
        for s in signals:
            st.write("- " + s)

        # ChatGPT 投资解读
        st.subheader("📝 ChatGPT 投资解读")
        report = ai_analysis(code, df, signals)
        st.write(report)

                      # --- 策略回测（form + session_state + 锚点跳转） ---
        st.subheader("📊 策略回测：MACD 金叉/死叉")

        with st.form("backtest_form"):
            col1, col2 = st.columns(2)
            with col1:
                lookback = st.number_input("回测天数 (lookback)", min_value=30, max_value=365, value=90, step=10)
            with col2:
                holding_days = st.number_input("持仓天数 (holding_days)", min_value=1, max_value=30, value=5, step=1)
            submitted = st.form_submit_button("运行回测")

        if submitted:
            results, trades = backtest_macd(df, lookback=lookback, holding_days=holding_days)
            st.session_state["backtest_results"] = results
            st.session_state["backtest_trades"] = trades
            st.session_state["lookback"] = lookback
            st.session_state["holding_days"] = holding_days
            # 提交后设置 URL 参数，刷新后会跳到锚点
            st.experimental_set_query_params(section="backtest")

        # 🚀 提交后刷新也能显示结果
        if "backtest_results" in st.session_state:
            # 定义一个锚点
            st.markdown("<a name='backtest'></a>", unsafe_allow_html=True)

            results = st.session_state["backtest_results"]
            trades = st.session_state["backtest_trades"]
            lookback = st.session_state["lookback"]
            holding_days = st.session_state["holding_days"]

            st.write(f"过去 {lookback} 天内：")
            st.write(f"- MACD 金叉次数: {results['金叉']['次数']}，{holding_days}日后上涨胜率: {results['金叉']['胜率']:.2%}")
            st.write(f"- MACD 死叉次数: {results['死叉']['次数']}，{holding_days}日后下跌胜率: {results['死叉']['胜率']:.2%}")

            if trades:
                st.write(f"最近几次交易回测记录 (持仓 {holding_days} 天)：")
                trade_df = pd.DataFrame(trades, columns=["信号", "日期", "买入价", "卖出价", "收益率"])
                trade_df["收益率"] = trade_df["收益率"].map(lambda x: f"{x:.2%}")
                st.dataframe(trade_df.tail(5))

                # 📥 下载按钮
                csv = trade_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="下载回测结果 CSV",
                    data=csv,
                    file_name=f"backtest_{code}.csv",
                    mime="text/csv"
                )
            else:
                st.info("⚠️ 最近没有检测到有效的 MACD 金叉/死叉信号，无法回测。")


