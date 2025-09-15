import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
from datetime import datetime

st.set_page_config(page_title="热门板块/ETF自动选股AI分析系统", layout="wide")
st.title("🔥 热门行业/概念/ETF池 批量信号+AI智能分析")

# --- 1. 板块和ETF数据自动加载 ---
@st.cache_data(ttl=3600)
def get_hot_industry_rank():
    return ak.stock_board_industry_rank_em()
@st.cache_data(ttl=3600)
def get_hot_concept_rank():
    return ak.stock_board_concept_rank_em()
@st.cache_data(ttl=3600)
def get_hot_board_members(board_type, board_name):
    if board_type == "industry":
        df = ak.stock_board_industry_cons_em(symbol=board_name)
    elif board_type == "concept":
        df = ak.stock_board_concept_cons_em(symbol=board_name)
    return df["代码"].tolist(), df

@st.cache_data
def get_all_etf_codes():
    df = ak.fund_etf_category_sina(symbol="ETF基金")
    return df["symbol"].tolist(), df

# --- 2. 板块/概念/ETF池选择 ---
tab1, tab2 = st.tabs(["🔥 板块/概念/ETF批量选股信号", "批量AI智能分析"])

with tab1:
    st.subheader("🔮 热门行业/概念/ETF池 自动信号筛选与明细解释")
    pool_type = st.selectbox(
        "请选择池类型", 
        ["热门行业板块", "热门概念板块", "全部ETF", "热门ETF（市值Top50）", "自定义股票池"], 
        index=0
    )
    # 自动拉取成分池
    code_pool, pool_show_df = [], None
    if pool_type == "热门行业板块":
        industry_df = get_hot_industry_rank()
        show_num = st.slider("显示前N个热门行业", 5, 30, 12)
        st.dataframe(industry_df[["板块名称","最新价","涨跌幅","领涨股"]].head(show_num), use_container_width=True)
        hot_blocks = industry_df["板块名称"].head(show_num).tolist()
        blocks_selected = st.multiselect("选择分析的热门行业板块", hot_blocks, default=hot_blocks[:2])
        for blk in blocks_selected:
            codes, df_blk = get_hot_board_members("industry", blk)
            code_pool += codes
            pool_show_df = pd.concat([pool_show_df, df_blk]) if pool_show_df is not None else df_blk
    elif pool_type == "热门概念板块":
        concept_df = get_hot_concept_rank()
        show_num = st.slider("显示前N个热门概念", 5, 30, 12)
        st.dataframe(concept_df[["板块名称","最新价","涨跌幅","领涨股"]].head(show_num), use_container_width=True)
        hot_blocks = concept_df["板块名称"].head(show_num).tolist()
        blocks_selected = st.multiselect("选择分析的热门概念板块", hot_blocks, default=hot_blocks[:2])
        for blk in blocks_selected:
            codes, df_blk = get_hot_board_members("concept", blk)
            code_pool += codes
            pool_show_df = pd.concat([pool_show_df, df_blk]) if pool_show_df is not None else df_blk
    elif pool_type == "全部ETF":
        etf_codes, etf_df = get_all_etf_codes()
        show_num = st.slider("显示前N个ETF", 20, 200, 60)
        st.dataframe(etf_df[["symbol","name","price"]].head(show_num), use_container_width=True)
        code_pool = etf_codes[:show_num]
        pool_show_df = etf_df.head(show_num)
    elif pool_type == "热门ETF（市值Top50）":
        etf_codes, etf_df = get_all_etf_codes()
        etf_df = etf_df.sort_values("amount", ascending=False)
        st.dataframe(etf_df[["symbol","name","price","amount"]].head(50), use_container_width=True)
        code_pool = etf_df["symbol"].head(50).tolist()
        pool_show_df = etf_df.head(50)
    else:
        code_input = st.text_area("手动输入代码（逗号、空格或换行均可）")
        code_pool = [c.strip() for c in code_input.replace('，', ',').replace('\n', ',').split(',') if c.strip()]

    st.info(f"本次选股池共 {len(code_pool)} 只标的。")
    if pool_show_df is not None:
        st.dataframe(pool_show_df[["代码" if "代码" in pool_show_df.columns else "symbol","名称" if "名称" in pool_show_df.columns else "name"]], use_container_width=True)
    start_date = st.date_input("起始日期", value=pd.to_datetime("2024-01-01"), key="blk_date")
    btn = st.button("一键批量信号筛选", key="blk_btn")

    # --- 技术信号检测逻辑同前，可复用 ---
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
        except Exception:
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

    # === 技术信号筛选主流程 ===
    if btn and len(code_pool) > 0:
        st.info(f"开始批量检测（建议单池不超300只）")
        result_table = []
        for i, code in enumerate(code_pool):
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
            if i < 5:
                st.markdown(f"#### 【{code}】选股信号：{'、'.join(signals) if signals else '无明显信号'}")
                with st.expander("信号检测明细（点击展开）", expanded=False):
                    for line in explain:
                        st.write(line)
        selected = [r for r in result_table if "无明显信号" not in r["信号"]]
        if selected:
            st.subheader("✅ 入选标的与信号（可导出）")
            df_sel = pd.DataFrame(selected)
            st.dataframe(df_sel[["代码","信号"]])
            st.download_button(
                "导出全部明细为Excel",
                data=pd.DataFrame(result_table).to_excel(index=False),
                file_name="选股明细.xlsx"
            )
        else:
            st.warning("暂无标的触发选股信号，可换池、调策略。")
    elif btn:
        st.warning("请先选择或输入池。")
    else:
        st.markdown("> 板块/概念/ETF/自定义一键批量信号检测、明细解释、AI批量扩展请切换到Tab2。")

# === TAB2: ETF/自定义池 批量AI技术点评 ===
with tab2:
    st.subheader("批量AI智能分析（支持ETF/概念/自定义池）")
    import plotly.graph_objects as go
    openai_key = st.text_input("OpenAI API KEY（AI点评/趋势预测）", type="password", key="ai_key")
    codes_input = st.text_area("输入待分析的股票/ETF代码（自动批量，逗号或换行分隔）", value="", key="ai_codes")
    trend_days = st.selectbox("AI预测未来天数", options=[1, 3, 5, 7], index=1, key="ai_trend_days")
    ai_enable = st.toggle("启用AI技术点评", value=True, key="ai_enable")
    btn_ai = st.button("批量AI分析", key="btn_ai")
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
你是一位A股/ETF专业量化分析师。以下是{code}最近60日的每日行情（日期,开盘,收盘,最高,最低,成交量），请根据技术走势、成交量变化，预测该标的未来{trend_days}日的涨跌趋势，并判断是否存在启动信号、买卖机会，请以精炼中文输出一份点评。数据如下（csv格式）：
{data_str}
"""
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一位专业A股/ETF分析师。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.6,
            )
            return chat_completion.choices[0].message.content
        except Exception as ex:
            return f"AI分析调用失败：{ex}"

    if btn_ai:
        codes = [c.strip() for c in codes_input.replace('，', ',').replace('\n', ',').split(',') if c.strip()]
        for code in codes:
            st.header(f"【{code}】AI分析")
            df = fetch_ak_data(code, pd.to_datetime("2024-01-01"))
            if df.empty:
                st.warning(f"{code} 数据未获取到。")
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
        st.markdown("> 批量支持ETF/板块/自选池AI技术点评、K线可视化，可输入多个代码。")

