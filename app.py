import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

st.set_page_config(page_title="A股批量智能技术分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量AI自动选股 & 智能趋势点评")

# 1. 指数成分股接口兼容多字段
@st.cache_data(show_spinner=False)
def get_index_codes(index_code):
    """
    拉取指数成分股列表，兼容字段新旧版本
    """
    try:
        df = ak.index_stock_cons(symbol=index_code)  # 关键点：必须用symbol参数
        if "con_code" in df.columns:
            return df["con_code"].tolist()
        elif "成分券代码" in df.columns:
            return df["成分券代码"].tolist()
        else:
            st.warning("指数成分股数据字段异常！")
            return []
    except Exception as e:
        st.warning(f"拉取指数成分股失败: {e}")
        return []

# 2. ETF池拉取
@st.cache_data(show_spinner=False)
def get_all_etf_codes():
    try:
        etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
        if "symbol" in etf_df.columns:
            return etf_df["symbol"].tolist()
        elif "代码" in etf_df.columns:
            return etf_df["代码"].tolist()
        else:
            st.warning("ETF数据结构异常！")
            return []
    except Exception as e:
        st.warning(f"ETF拉取失败: {e}")
        return []

# 3. 全A股（不含沪深300、科创50）
@st.cache_data(show_spinner=False)
def get_all_a_codes_exclude_indexes():
    """
    拉取A股全市场代码，并自动剔除沪深300、科创50成分股（如你需要）
    """
    try:
        all_a = ak.stock_info_a_code_name()["code"].tolist()
        hs300 = set(get_index_codes("000300"))
        kc50 = set(get_index_codes("000688"))
        only_a = [x for x in all_a if x not in hs300 and x not in kc50]
        return only_a
    except Exception as e:
        st.warning(f"A股拉取失败: {e}")
        return []

# 4. 热门概念板块拉取兼容
@st.cache_data(ttl=300, show_spinner=False)
def get_hot_concept_boards(topn=20):
    try:
        df = ak.stock_board_concept_name_ths()
        # 字段自动识别（兼容旧的“概念名称”和“涨幅”）
        name_col = "板块名称" if "板块名称" in df.columns else ("概念名称" if "概念名称" in df.columns else None)
        pct_col = "涨跌幅" if "涨跌幅" in df.columns else ("涨幅" if "涨幅" in df.columns else None)
        if name_col and pct_col:
            hot_df = df.sort_values(pct_col, ascending=False).head(topn)
            return hot_df[[name_col, pct_col]]
        else:
            st.warning("热门概念板块数据字段异常！")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"拉取热门板块失败: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_board_stocks(board_name):
    try:
        df = ak.stock_board_concept_cons_ths(symbol=board_name)
        if "代码" in df.columns:
            return df["代码"].tolist()
        elif "股票代码" in df.columns:
            return df["股票代码"].tolist()
        else:
            return []
    except Exception:
        return []

# ========= 数据与指标部分省略，复用你的原有逻辑 ===========

# fetch_ak_data, calc_indicators, signal_with_explain, ai_trend_report等函数和原来一致
# ... 复制你的原有实现即可 ...

# ========= 主界面 =========
tab1, tab2 = st.tabs(["🪄 批量自动选股(分批)", "个股批量分析+AI点评"])

with tab1:
    st.subheader("全A股/沪深300/科创50/ETF/热门板块自动选股，趋势股池专属板块")
    market_pool = st.selectbox(
        "选择批量选股池",
        options=["全A股(不含沪深300、科创50)", "沪深300", "科创50", "全ETF", "热门概念板块", "自定义"],
        index=0
    )
    codes = []
    show_desc = ""
    if market_pool.startswith("全A股"):
        codes = get_all_a_codes_exclude_indexes()
        show_desc = "（全A股，已剔除沪深300与科创50成分股）"
    elif market_pool == "沪深300":
        codes = get_index_member_codes("000300")
        show_desc = "（沪深300指数成分股）"
    elif market_pool == "科创50":
        codes = get_index_member_codes("000688")
        show_desc = "（科创50指数成分股）"
    elif market_pool == "全ETF":
        codes = get_all_etf_codes()
        show_desc = "（全ETF基金）"
    elif market_pool == "热门概念板块":
        st.markdown("#### 🔥 今日热门概念板块排行（涨幅前20）")
        hot_boards = get_hot_concept_boards()
        if not hot_boards.empty:
            st.dataframe(hot_boards, hide_index=True, use_container_width=True)
            selected_boards = st.multiselect("选择要检测的热门板块（可多选）", hot_boards["板块名称"].tolist())
            for board in selected_boards:
                codes += get_board_stocks(board)
            show_desc = "（热门概念板块池，成分股合并）"
        else:
            st.warning("未能获取热门板块数据")
    else:
        codes_input = st.text_area("手动输入代码（逗号、空格或换行均可）")
        for line in codes_input.splitlines():
            for c in line.replace('，', ',').replace(' ', ',').split(','):
                if c.strip():
                    codes.append(c.strip())
        show_desc = "（自定义池）"
    codes = list(set(codes))
    st.info(f"{show_desc} 选股池共计 {len(codes)} 只标的。")

    # ======== 分批分页 ========
    BATCH_SIZE = 200
    if "page" not in st.session_state:
        st.session_state["page"] = 0
    total_batches = (len(codes) - 1) // BATCH_SIZE + 1 if codes else 1
    current_batch = st.session_state["page"]
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("上一批", disabled=(current_batch == 0)):
            st.session_state["page"] = max(0, current_batch - 1)
    with col2:
        st.write(f"第{current_batch+1}/{total_batches}批")
    with col3:
        if st.button("下一批", disabled=(current_batch+1 >= total_batches)):
            st.session_state["page"] = min(total_batches - 1, current_batch + 1)
    codes_this_batch = codes[current_batch*BATCH_SIZE : (current_batch+1)*BATCH_SIZE]
    st.info(f"本批共{len(codes_this_batch)}只，股票池共{len(codes)}只。")

    start_date = st.date_input("起始日期", value=pd.to_datetime("2024-01-01"), key="pick_start")
    openai_key = st.text_input("如需AI批量趋势点评，请输入OpenAI KEY", type="password", key="tab1_ai_key")
    ai_batch = st.toggle("批量AI智能点评", value=True, key="ai_batch_tab1")
    trend_days = st.selectbox("AI预测未来天数", options=[1, 3, 5, 7], index=1, key="tab1_trend_days")
    btn = st.button("本批次一键自动选股", key="btn_pick")

    # ========= 多线程并发拉取K线 + 信号AI智能点评 =========
    if btn and codes_this_batch:
        st.info("开始本批次数据分析…")
        result_table = []
        prog = st.progress(0, text="数据处理中…")
        def fetch_ak_data_safe(code, start_date):
            try:
                df = fetch_ak_data(code, start_date)
                return code, df
            except Exception as e:
                return code, pd.DataFrame()
        result_dict = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_code = {executor.submit(fetch_ak_data_safe, code, start_date): code for code in codes_this_batch}
            for i, future in enumerate(as_completed(future_to_code)):
                code, df = future.result()
                result_dict[code] = df
                prog.progress((i+1)/len(codes_this_batch), text=f"拉取进度：{i+1}/{len(codes_this_batch)}")
        prog.empty()

        # 2. 信号判别和AI点评（仅对有信号股AI）
        trend_stock_pool = []
        for i, code in enumerate(codes_this_batch):
            df = result_dict.get(code, pd.DataFrame())
            if df.empty or len(df) < 25:
                continue
            df = calc_indicators(df)
            signals, explain = signal_with_explain(df)
            ai_result = ""
            if ai_batch and openai_key and signals:
                with st.spinner(f"AI分析{code}中..."):
                    ai_result = ai_trend_report(df, code, trend_days, openai_key)
                    time.sleep(1.2)  # 限速，防止被封
            row = {
                "代码": code,
                "信号": "、".join(signals) if signals else "无明显信号",
                "明细解释": "\n".join(explain),
                "AI点评": ai_result
            }
            result_table.append(row)
            if signals:
                trend_stock_pool.append(row)
            if i < 6 and signals:
                st.markdown(f"#### 【{code}】选股信号：{'、'.join(signals)}")
                with st.expander("信号检测明细（点击展开）", expanded=False):
                    for line in explain:
                        st.write(line)
                if ai_result:
                    st.info(f"AI点评：{ai_result}")

        # ========= 趋势股池专属板块 =========
        st.markdown("---")
        st.subheader("🚩 本批趋势股池（所有检测出信号的标的）")
        if trend_stock_pool:
            df_pool = pd.DataFrame(trend_stock_pool)
            st.dataframe(df_pool[["代码", "信号", "AI点评"]], use_container_width=True)
            # 支持导出趋势股池
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_pool.to_excel(writer, index=False)
            st.download_button(
                "导出趋势股池为Excel",
                data=output.getvalue(),
                file_name="趋势股池.xlsx"
            )
        else:
            st.info("本批暂无趋势信号股。")

        # ========= 全部结果导出 =========
        st.markdown("---")
        with st.expander("导出全部明细（含无信号）", expanded=False):
            df_all = pd.DataFrame(result_table)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_all.to_excel(writer, index=False)
            st.download_button(
                "导出全部分析明细为Excel",
                data=output.getvalue(),
                file_name="AI选股明细_全部含无信号.xlsx"
            )
    else:
        st.markdown("> 支持全A股（已剔除沪深300/科创50）、沪深300、科创50、ETF、热门板块一键分批自动选股+AI趋势点评。趋势信号股高亮输出专属板块。")

# 其余Tab2同前，略。
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
