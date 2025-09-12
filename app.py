import streamlit as st
import pandas as pd
import pandas_ta as ta
import openai
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI, RateLimitError, AuthenticationError, OpenAIError
from st_aggrid import AgGrid, GridOptionsBuilder   # ✅ 引入 AgGrid

# ✅ 设置 OpenAI 密钥
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ✅ 页面初始化
st.set_page_config(page_title="AkShare + ChatGPT 股票分析", layout="wide")
st.title("📈 AkShare + ChatGPT 技术面股票分析")

# ………………………………（前面的 fetch_ak_kline、analyze_tech、backtest_signals、explain_by_gpt 函数保持不变）………………………………


# ✅ 用户输入股票代码
stock_code = st.text_input("请输入股票代码（6位，不带 SH/SZ 后缀）如 600519:")

if stock_code:
    with st.spinner("正在获取数据和分析中..."):
        df = fetch_ak_kline(stock_code)
        if df.empty:
            st.stop()

        df = analyze_tech(df)
        last_row = df.iloc[-1]

        # ✅ 最近数据表（用 AgGrid）
        st.subheader("📊 最近行情与技术指标（交互表格）")
        recent_df = df.tail(30)[['date','open','high','low','close','volume','MACD','MACD_signal','RSI']]

        gb = GridOptionsBuilder.from_dataframe(recent_df)
        gb.configure_pagination(paginationAutoPageSize=True)  # 分页
        gb.configure_side_bar()                               # 侧边栏：筛选/隐藏列
        gb.configure_default_column(editable=False, groupable=True, resizable=True,
                                    sortable=True, filter=True)
        grid_options = gb.build()

        AgGrid(recent_df, gridOptions=grid_options, enable_enterprise_modules=True,
               height=400, fit_columns_on_grid_load=True)

        # ✅ 图表可视化
        st.subheader("📉 图表分析展示")
        chart_tab = st.tabs(["K线+均线+BOLL+成交量", "MACD", "RSI 相对强弱指标"])

        # ……（这里的 K 线、MACD、RSI 图表代码保持不变）……

        # ✅ 策略回测结果
        st.subheader("📈 策略信号回测结果")

        hold_days = st.slider("选择回测持有天数", min_value=3, max_value=20, value=5, step=1)
        backtest_result = backtest_signals(df, hold_days=hold_days)
        st.write(f"买入信号样本数: {backtest_result['样本数']}")
        st.write(f"平均 {hold_days} 日涨幅: {backtest_result['平均涨幅']}%")
        st.write(f"胜率: {backtest_result['胜率']}%")

        # ✅ 多周期对比表格（用 AgGrid）
        st.markdown("### 📊 多周期对比（3 / 5 / 10 / 20 日）")
        compare_days = [3, 5, 10, 20]
        results = []
        for d in compare_days:
            res = backtest_signals(df, hold_days=d)
            results.append([d, res['样本数'], f"{res['平均涨幅']}%", f"{res['胜率']}%"])
        compare_df = pd.DataFrame(results, columns=["持有天数","样本数","平均涨幅","胜率"])

        gb2 = GridOptionsBuilder.from_dataframe(compare_df)
        gb2.configure_default_column(editable=False, groupable=True, resizable=True,
                                     sortable=True, filter=True)
        grid_options2 = gb2.build()

        AgGrid(compare_df, gridOptions=grid_options2, enable_enterprise_modules=True,
               height=300, fit_columns_on_grid_load=True)

        # ✅ 柱状图对比
        st.markdown("### 📊 多周期可视化对比")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=compare_df["持有天数"], 
                                 y=[float(v.strip('%')) for v in compare_df["平均涨幅"]],
                                 name="平均涨幅(%)"))
        fig_bar.add_trace(go.Bar(x=compare_df["持有天数"], 
                                 y=[float(v.strip('%')) for v in compare_df["胜率"]],
                                 name="胜率(%)"))
        fig_bar.update_layout(barmode='group', height=400, margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

        # ✅ GPT 策略建议
        st.subheader("🧠 ChatGPT 策略建议")
        suggestion = explain_by_gpt(stock_code, last_row)
        st.markdown(suggestion)

else:
    st.info("请输入6位股票代码，例如 000001 或 600519")
