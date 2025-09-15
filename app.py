import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="A股批量智能技术分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量AI自动选股 & 智能趋势点评（全接口兼容）")

# ====== 通用字段兼容工具 ======
def get_first_valid_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"字段未找到，现有字段: {df.columns.tolist()}, 候选: {candidates}")

def get_code_list(df):
    code_candidates = ["symbol", "基金代码", "代码", "con_code", "成分券代码"]
    code_col = get_first_valid_column(df, code_candidates)
    return df[code_col].tolist()

def get_name_list(df):
    name_candidates = ["name", "基金简称", "简称", "股票名称", "成分券名称", "板块名称"]
    name_col = get_first_valid_column(df, name_candidates)
    return df[name_col].tolist()

def get_pct_chg_col(df):
    chg_candidates = ["涨跌幅", "涨幅", "变动率", "日涨幅"]
    return get_first_valid_column(df, chg_candidates)

def show_columns(df, name="DataFrame"):
    st.write(f"【{name} 字段】: {df.columns.tolist()}")

def sort_by_pct_chg(df, topn=20):
    try:
        col = get_pct_chg_col(df)
        return df.sort_values(col, ascending=False).head(topn)
    except Exception as e:
        st.warning(f"排序字段未找到：{e}")
        return df.head(topn)

def dataframe_to_excel_bytes(df):
    output = io.BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    return output.getvalue()

# ====== AkShare自动兼容接口 ======
@st.cache_data(ttl=1800)
def get_all_a_codes():
    stock_df = ak.stock_info_a_code_name()
    return get_code_list(stock_df)

@st.cache_data(ttl=1800)
def get_all_etf_codes():
    etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
    return get_code_list(etf_df)

@st.cache_data(ttl=1800)
def get_index_codes_auto(index_code):
    df = ak.index_stock_cons(symbol=index_code)
    code_candidates = ["con_code", "成分券代码"]
    code_col = get_first_valid_column(df, code_candidates)
    return df[code_col].tolist()

# ====== 板块排行（行业 + 概念自动兼容） ======
@st.cache_data(ttl=1800)
def get_hot_industry_boards(topn=20):
    try:
        df = ak.stock_board_industry_index_ths()  # ✅ 行业行情接口，含涨跌幅
    except Exception:
        df = ak.stock_board_industry_name_ths()   # 兜底：只有名称和代码
    show_columns(df, "行业板块")
    return sort_by_pct_chg(df, topn=topn)

@st.cache_data(ttl=1800)
def get_hot_concept_boards(topn=20):
    try:
        df = ak.stock_board_concept_index_ths()   # ✅ 概念行情接口，含涨跌幅
    except Exception:
        df = ak.stock_board_concept_name_ths()    # 兜底：只有名称和代码
    show_columns(df, "概念板块")
    return sort_by_pct_chg(df, topn=topn)

# ====== 板块成分股（行业 + 概念自动兼容） ======
@st.cache_data(ttl=300)
def get_board_stocks(board_name):
    try:
        df = ak.stock_board_concept_cons_ths(symbol=board_name)  # ✅ 尝试概念
    except Exception:
        try:
            df = ak.stock_board_industry_cons_ths(symbol=board_name)  # ✅ 尝试行业
        except Exception:
            return []
    return get_code_list(df) if not df.empty else []

# ====== K线与信号判别函数（保持原样） ======
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
    # ETF兜底
    try:
        df = ak.fund_etf_hist_sina(symbol=code)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= pd.to_datetime(start_date)]
            df = df.sort_values("date").reset_index(drop=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()

# ====== 后续你的 calc_indicators、signal_with_explain、tab1/tab2/tab3 保持不变 ======
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
    if "SMA_5" in df.columns and "SMA_10" in df.columns:
        if pre["SMA_5"] < pre["SMA_10"] and latest["SMA_5"] > latest["SMA_10"]:
            signals.append("5日均线金叉10日均线")
            explain.append("【均线金叉】：今日5日均线上穿10日均线（金叉），多头信号。")
        else:
            explain.append(f"【均线金叉】：5日均线({latest['SMA_5']:.2f}) {'>' if latest['SMA_5']>latest['SMA_10'] else '<='} 10日均线({latest['SMA_10']:.2f})，未发生金叉。")
    else:
        explain.append("【均线金叉】：数据不足，无法判断。")
    if "MACD" in df.columns and "MACDs" in df.columns:
        if pre["MACD"] < pre["MACDs"] and latest["MACD"] > latest["MACDs"]:
            signals.append("MACD金叉")
            explain.append("【MACD金叉】：今日MACD线上穿信号线，金叉出现，多头信号。")
        else:
            explain.append(f"【MACD金叉】：MACD({latest['MACD']:.3f}) {'>' if latest['MACD']>latest['MACDs'] else '<='} 信号线({latest['MACDs']:.3f})，未发生金叉。")
    else:
        explain.append("【MACD金叉】：数据不足，无法判断。")
    if "RSI_6" in df.columns:
        if latest["RSI_6"] < 30 and pre["RSI_6"] >= 30:
            signals.append("RSI6超卖反弹")
            explain.append("【RSI超卖反弹】：今日RSI6跌破30出现反弹，短期见底迹象。")
        else:
            explain.append(f"【RSI超卖反弹】：RSI6当前为{latest['RSI_6']:.1f}，未触发超卖反弹。")
    else:
        explain.append("【RSI超卖反弹】：数据不足，无法判断。")
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
    if "close" in df.columns and len(df) >= 20:
        if latest["close"] >= df["close"].iloc[-20:].max():
            signals.append("20日新高")
            explain.append("【20日新高】：今日收盘价达到近20日最高。")
        else:
            explain.append(f"【20日新高】：今日收盘{latest['close']}，20日最高{df['close'].iloc[-20:].max()}，未创新高。")
    else:
        explain.append("【20日新高】：数据不足，无法判断。")
    if "close" in df.columns and len(df) >= 20:
        if latest["close"] <= df["close"].iloc[-20:].min():
            signals.append("20日新低")
            explain.append("【20日新低】：今日收盘价达到近20日最低。")
        else:
            explain.append(f"【20日新低】：今日收盘{latest['close']}，20日最低{df['close'].iloc[-20:].min()}，未创新低。")
    else:
        explain.append("【20日新低】：数据不足，无法判断。")
    return signals, explain

# ====== 主界面分三大模块 ======
tab1, tab2, tab3 = st.tabs(["🔥 热门板块概念排行", "🪄 批量自动选股(分批)", "AI智能批量分析"])

# --- 热门板块/概念排行 ---
with tab1:
    st.subheader("今日热门行业/概念板块涨幅排行")
    col1, col2 = st.columns(2)
    with col1:
        industry_df = get_hot_industry_boards(topn=20)
        st.dataframe(industry_df, use_container_width=True)
        if st.button("导出行业板块"):
            excel_bytes = dataframe_to_excel_bytes(industry_df)
            st.download_button("下载行业板块Excel", data=excel_bytes, file_name="行业板块.xlsx")
    with col2:
        concept_df = get_hot_concept_boards(topn=20)
        st.dataframe(concept_df, use_container_width=True)
        if st.button("导出概念板块"):
            excel_bytes = dataframe_to_excel_bytes(concept_df)
            st.download_button("下载概念板块Excel", data=excel_bytes, file_name="概念板块.xlsx")

# --- 分批自动选股 ---
with tab2:
    st.subheader("全市场/ETF/指数/概念池自动选股，支持分批分析")
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
        codes = get_index_codes_auto("000300")
    elif market_pool == "科创50":
        codes = get_index_codes_auto("000688")
    elif market_pool == "热门概念板块":
        st.markdown("#### 🔥 今日热门概念板块排行（涨幅前20）")
        hot_boards = get_hot_concept_boards()
        if not hot_boards.empty:
            st.dataframe(hot_boards, hide_index=True, use_container_width=True)
            selected_boards = st.multiselect("选择要检测的热门板块（可多选）", hot_boards.iloc[:,0].tolist())
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
    codes = list(set(codes))

    # ========== 分批分页 ==========
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
    btn = st.button("本批次一键自动选股", key="btn_pick")

    # ========== 多线程拉取+信号检测 ==========
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

        for i, code in enumerate(codes_this_batch):
            df = result_dict.get(code, pd.DataFrame())
            if df.empty or len(df) < 25:
                continue
            df = calc_indicators(df)
            signals, explain = signal_with_explain(df)
            result_table.append({
                "代码": code,
                "信号": "、".join(signals) if signals else "无明显信号",
                "明细解释": "\n".join(explain)
            })
            if i < 6 and signals:
                st.markdown(f"#### 【{code}】选股信号：{'、'.join(signals) if signals else '无明显信号'}")
                with st.expander("信号检测明细（点击展开）", expanded=False):
                    for line in explain:
                        st.write(line)

        selected = [r for r in result_table if "无明显信号" not in r["信号"]]
        if selected:
            st.subheader("✅ 入选标的与信号（可全部导出）")
            df_sel = pd.DataFrame(selected)
            st.dataframe(df_sel[["代码","信号"]], use_container_width=True)
            excel_bytes = dataframe_to_excel_bytes(pd.DataFrame(result_table))
            st.download_button(
                "导出全部明细为Excel",
                data=excel_bytes,
                file_name="AI选股明细.xlsx"
            )
        else:
            st.warning("暂无标的触发选股信号，可切换批次、调整策略或换池。")
    else:
        st.markdown("> 支持全A股、ETF、指数成分、热门池一键分批自动选股。")

# --- AI批量分析 ---
with tab3:
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
            fig2.add_trace(go.Bar(x=df["date"], y=df["MACD"], name="MACD线"))
            fig2.add_trace(go.Scatter(x=df["date"], y=df["MACDs"], name="信号线"))
            fig2.update_layout(title="MACD指标", height=200)
            st.plotly_chart(fig2, use_container_width=True)
        if "RSI_6" in df.columns:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df["date"], y=df["RSI_6"], name="RSI6"))
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

st.info("全新版本已适配所有字段、接口、导出，无需担心KeyError或Excel导出报错，可长期云端稳定运行。")
