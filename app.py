import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="A股批量分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量智能技术分析 & AI趋势预测")

tab1, tab2 = st.tabs(["🪄 全市场自动选股信号", "个股批量分析+AI点评"])

# =============== TAB 1：全A股/ETF/板块自动选股 ===============
with tab1:
    st.subheader("全市场池自动加载+多策略选股信号检测")
    # --- 选股池选择 ---
    @st.cache_data
    def get_all_a_codes():
        stock_df = ak.stock_info_a_code_name()
        return stock_df["code"].tolist()
    @st.cache_data
    def get_all_etf_codes():
        etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
        return etf_df["symbol"].tolist()
    @st.cache_data
    def get_index_codes(index_code):
        df = ak.index_stock_cons(index=index_code)
        return df["con_code"].tolist()
    @st.cache_data(ttl=300)
    def get_hot_concept_boards(topn=20):
        try:
            df = ak.stock_board_concept_name_ths()
            hot_df = df.sort_values("涨跌幅", ascending=False).head(topn)
            return hot_df[["板块名称", "涨跌幅"]]
        except:
            return pd.DataFrame()
    @st.cache_data(ttl=300)
    def get_board_stocks(board_name):
        try:
            df = ak.stock_board_concept_cons_ths(symbol=board_name)
            return df["代码"].tolist()
        except:
            return []

    market_pool = st.selectbox(
        "选择批量选股池",
        options=["全A股", "全ETF", "沪深300", "科创50", "热门概念板块", "自定义"],
        index=0
    )
    codes = []
    if market_pool == "全A股":
        codes = get_all_a_codes()
    elif market_pool == "全ETF":
        codes = get_all_etf_codes()
    elif market_pool == "沪深300":
        codes = get_index_codes("000300")
    elif market_pool == "科创50":
        codes = get_index_codes("000688")
    elif market_pool == "热门概念板块":
        st.markdown("#### 🔥 今日热门概念板块排行（涨幅前20）")
        hot_boards = get_hot_concept_boards()
        if not hot_boards.empty:
            st.dataframe(hot_boards, hide_index=True, use_container_width=True)
            selected_boards = st.multiselect("选择要检测的热门板块（可多选）", hot_boards["板块名称"].tolist())
            for board in selected_boards:
                codes += get_board_stocks(board)
        else:
            st.warning("未能获取热门板块数据")
    else:
        codes_input = st.text_area("手动输入代码（逗号、空格或换行均可）")
        for line in codes_input.splitlines():
            for c in line.replace('，', ',').replace(' ', ',').split(','):
                if c.strip():
                    codes.append(c.strip())
    codes = list(set(codes))  # 去重
    st.info(f"本次选股池共计 {len(codes)} 只标的。")
    start_date = st.date_input("起始日期", value=pd.to_datetime("2024-01-01"), key="pick_start")
    btn = st.button("一键批量自动选股", key="btn_pick")

    def fetch_ak_data(code, start_date):
        df = pd.DataFrame()
        try:
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

    def calc_indicators(df):
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
            df["RSI_6"] = ta.rsi(df["close"], length=6)
        except Exception as e:
            pass
        return df

    def signal_with_explain(df):
        explain = []
        signals = []
        latest = df.iloc[-1]
        pre = df.iloc[-2] if len(df) >= 2 else latest

        # 1. 均线金叉
        if "SMA_5" in df.columns and "SMA_10" in df.columns:
            if pre["SMA_5"] < pre["SMA_10"] and latest["SMA_5"] > latest["SMA_10"]:
                signals.append("5日均线金叉10日均线")
                explain.append("【均线金叉】：今日5日均线上穿10日均线（金叉），多头信号。")
            else:
                explain.append(f"【均线金叉】：5日均线({latest['SMA_5']:.2f}) {'>' if latest['SMA_5']>latest['SMA_10'] else '<='} 10日均线({latest['SMA_10']:.2f})，未发生金叉。")
        else:
            explain.append("【均线金叉】：数据不足，无法判断。")

        # 2. MACD金叉
        if "MACD" in df.columns and "MACDs" in df.columns:
            if pre["MACD"] < pre["MACDs"] and latest["MACD"] > latest["MACDs"]:
                signals.append("MACD金叉")
                explain.append("【MACD金叉】：今日MACD线上穿信号线，金叉出现，多头信号。")
            else:
                explain.append(f"【MACD金叉】：MACD({latest['MACD']:.3f}) {'>' if latest['MACD']>latest['MACDs'] else '<='} 信号线({latest['MACDs']:.3f})，未发生金叉。")
        else:
            explain.append("【MACD金叉】：数据不足，无法判断。")

        # 3. RSI超卖反弹
        if "RSI_6" in df.columns:
            if latest["RSI_6"] < 30 and pre["RSI_6"] >= 30:
                signals.append("RSI6超卖反弹")
                explain.append("【RSI超卖反弹】：今日RSI6跌破30出现反弹，短期见底迹象。")
            else:
                explain.append(f"【RSI超卖反弹】：RSI6当前为{latest['RSI_6']:.1f}，未触发超卖反弹。")
        else:
            explain.append("【RSI超卖反弹】：数据不足，无法判断。")

        # 4. 放量突破
        if "volume" in df.columns and "close" in df.columns and len(df) >= 6:
            pre_vol = df["volume"].iloc[-6:-1].mean()
            vol_up = latest["volume"] > 1.5 * pre_vol
            pct_chg = (latest["close"] - pre["close"]) / pre["close"] * 100 if pre["close"] > 0 else 0
            if vol_up and pct_chg > 2:
                signals.append("放量突破")
                explain.append("【放量突破】：今日成交量明显放大，且涨幅超过2%，主力资金有启动迹象。")
            else:
                explain.append(f"【放量突破】：今日成交量{latest['volume']}，均量{pre_vol:.0f}，{'放量' if vol_up else '未放量'}，涨幅{pct_chg:.2f}%。")
        else:
            explain.append("【放量突破】：数据不足，无法判断。")

        # 5. 20日新高
        if "close" in df.columns and len(df) >= 20:
            if latest["close"] >= df["close"].iloc[-20:].max():
                signals.append("20日新高")
                explain.append("【20日新高】：今日收盘价达到近20日最高。")
            else:
                explain.append(f"【20日新高】：今日收盘{latest['close']}，20日最高{df['close'].iloc[-20:].max()}，未创新高。")
        else:
            explain.append("【20日新高】：数据不足，无法判断。")

        # 6. 20日新低
        if "close" in df.columns and len(df) >= 20:
            if latest["close"] <= df["close"].iloc[-20:].min():
                signals.append("20日新低")
                explain.append("【20日新低】：今日收盘价达到近20日最低。")
            else:
                explain.append(f"【20日新低】：今日收盘{latest['close']}，20日最低{df['close'].iloc[-20:].min()}，未创新低。")
        else:
            explain.append("【20日新低】：数据不足，无法判断。")

        return signals, explain

    if btn:
        st.info(f"开始批量检测，请耐心等待（建议每次选股池不超过200只，太多易超时）")
        result_table = []
        for i, code in enumerate(codes):
            df = fetch_ak_data(code, start_date)
            if df.empty or len(df) < 25:
                continue
            df = calc_indicators(df)
            signals, explain = signal_with_explain(df)
            result_table.append({
                "代码": code,
                "信号": "、".join(signals) if signals else "无明显信号",
                "明细解释": "\n".join(explain)
            })
            if i < 10:
                st.markdown(f"#### 【{code}】选股信号：{'、'.join(signals) if signals else '无明显信号'}")
                with st.expander("信号检测明细（点击展开）", expanded=False):
                    for line in explain:
                        st.write(line)
        selected = [r for r in result_table if "无明显信号" not in r["信号"]]
        if selected:
            st.subheader("✅ 入选标的与信号（可全部导出）")
            df_sel = pd.DataFrame(selected)
            st.dataframe(df_sel[["代码","信号"]])
            st.download_button(
                "导出全部明细为Excel",
                data=pd.DataFrame(result_table).to_excel(index=False),
                file_name="选股明细.xlsx"
            )
        else:
            st.warning("暂无标的触发选股信号，可调整策略或换池。")
    else:
        st.markdown("> 支持全A股、ETF、指数成分、热门池一键批量自动选股+明细解释，结果可导出。")

# =============== TAB 2：原有批量分析&AI趋势点评 ===============
with tab2:
    st.subheader("自定义股票池批量分析+AI智能点评")
    openai_key = st.text_input("请输入你的OpenAI API KEY（用于AI点评/趋势预测）", type="password", key="ai_key")
    codes_input = st.text_input("请输入A股股票代码（支持批量,如 600519,000977,588170）：", value="000977,518880", key="ai_codes")
    start_date = st.date_input("选择起始日期", value=datetime.now().replace(year=2025, month=9, day=4), key="ai_date")
    ai_enable = st.toggle("启用AI趋势点评", value=True, key="ai_toggle")
    trend_days = st.selectbox("AI预测未来天数", options=[1, 3, 5, 7], index=1, key="ai_trend_days")
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
        if "MACD" in df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df["date"], y=df["MACDh"], name="MACD柱"))
            fig2.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD线"))
            fig2.add_trace(go.Scatter(x=df["date"], y=df["MACDs"], name="信号线"))
            fig2.update_layout(title="MACD指标", height=200)
            st.plotly_chart(fig2, use_container_width=True)
        if "RSI_6" in df.columns:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df["date"], y=df["RSI_6"], name="RSI6"))
            fig3.add_trace(go.Scatter(x=df["date"], y=df["RSI_12"], name="RSI12"))
            fig3.update_layout(title="RSI指标", height=200, yaxis=dict(range=[0,100]))
            st.plotly_chart(fig3, use_container_width=True)
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

    if st.button("批量分析", key="ai_btn"):
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
