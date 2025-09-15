import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="A股批量智能技术分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量AI自动选股 & 智能趋势点评")

# ========== AkShare字段自动适配 + 热门排行榜（万能工具） ==========
FIELD_MAP = {
    'code': ['代码', '股票代码', '证券代码', 'con_code', '基金代码', 'symbol'],
    'name': ['名称', '股票名称', '证券名称', '成分券名称', '基金简称', 'display_name', 'name'],
    'change_pct': ['涨跌幅', '涨幅', '涨跌幅度', 'changepercent', '涨幅(%)'],
    'board': ['板块名称', '概念名称', '行业名称'],
}
def get_col(df, key, strict=True, verbose=False):
    candidates = FIELD_MAP.get(key, [key])
    for cand in candidates:
        for col in df.columns:
            if cand.strip().lower() == col.strip().lower():
                if verbose:
                    print(f"找到{key}字段: {col}")
                return col
    if not strict:
        return None
    raise KeyError(f"DataFrame找不到‘{key}’相关字段，实际字段为: {df.columns.tolist()}")

@st.cache_data(show_spinner=False)
def get_all_a_codes():
    stock_df = ak.stock_info_a_code_name()
    code_col = get_col(stock_df, 'code')
    return stock_df[code_col].tolist()

@st.cache_data(show_spinner=False)
def get_all_etf_codes():
    etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
    code_col = get_col(etf_df, 'code')
    return etf_df[code_col].tolist()

@st.cache_data(show_spinner=False)
def get_index_codes(index_code):
    df = ak.index_stock_cons(symbol=index_code)
    code_col = get_col(df, 'code')
    return df[code_col].tolist()

@st.cache_data(ttl=300, show_spinner=False)
def get_hot_industry_boards(topn=20):
    try:
        df = ak.stock_board_industry_name_ths()
        change_col = get_col(df, 'change_pct', strict=False)
        board_col = get_col(df, 'board', strict=False)
        if change_col:
            hot_df = df.sort_values(change_col, ascending=False).head(topn)
        else:
            hot_df = df.head(topn)
        if board_col and change_col:
            return hot_df[[board_col, change_col]]
        elif board_col:
            return hot_df[[board_col]]
        else:
            return hot_df
    except Exception as e:
        print(f"获取行业板块异常: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_hot_concept_boards(topn=20):
    try:
        df = ak.stock_board_concept_name_ths()
        change_col = get_col(df, 'change_pct', strict=False)
        board_col = get_col(df, 'board', strict=False)
        if change_col:
            hot_df = df.sort_values(change_col, ascending=False).head(topn)
        else:
            hot_df = df.head(topn)
        if board_col and change_col:
            return hot_df[[board_col, change_col]]
        elif board_col:
            return hot_df[[board_col]]
        else:
            return hot_df
    except Exception as e:
        print(f"获取概念板块异常: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_hot_etf_boards(topn=20):
    try:
        df = ak.fund_etf_spot_em()
        change_col = get_col(df, 'change_pct', strict=False)
        name_col = get_col(df, 'name', strict=False)
        code_col = get_col(df, 'code', strict=False)
        if change_col:
            hot_df = df.sort_values(change_col, ascending=False).head(topn)
        else:
            hot_df = df.head(topn)
        cols = [c for c in [code_col, name_col, change_col] if c]
        return hot_df[cols] if cols else hot_df
    except Exception as e:
        print(f"获取ETF排行异常: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_board_stocks(board_name):
    try:
        df = ak.stock_board_concept_cons_ths(symbol=board_name)
        code_col = get_col(df, 'code')
        return df[code_col].tolist()
    except Exception as e:
        print(f"获取板块成分股异常: {e}")
        return []

# ============ 数据与指标函数 ============
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

def ai_trend_report(df, code, trend_days, openai_key):
    if not openai_key:
        return ""
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

# ========== 热门排行榜可视化（建议放在tab顶部） ==========
with st.expander("🚀 热门行业/概念/ETF排行榜（涨幅Top20）", expanded=False):
    st.markdown("#### 行业板块涨幅排行")
    ind_df = get_hot_industry_boards(topn=20)
    if not ind_df.empty:
        st.dataframe(ind_df, use_container_width=True)
    else:
        st.warning("未获取到行业板块数据")
    st.markdown("#### 概念板块涨幅排行")
    con_df = get_hot_concept_boards(topn=20)
    if not con_df.empty:
        st.dataframe(con_df, use_container_width=True)
    else:
        st.warning("未获取到概念板块数据")
    st.markdown("#### ETF基金涨幅排行")
    etf_df = get_hot_etf_boards(topn=20)
    if not etf_df.empty:
        st.dataframe(etf_df, use_container_width=True)
    else:
        st.warning("未获取到ETF数据")

# ============ 分批分页主界面 ============
tab1, tab2 = st.tabs(["🪄 批量自动选股(分批)", "个股批量分析+AI点评"])

with tab1:
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
        codes = get_index_codes("000300")
    elif market_pool == "科创50":
        codes = get_index_codes("000688")
    elif market_pool == "热门概念板块":
        st.markdown("#### 🔥 今日热门概念板块排行（涨幅前20）")
        hot_boards = get_hot_concept_boards()
        if not hot_boards.empty:
            # 自动适配字段展示
            st.dataframe(hot_boards, hide_index=True, use_container_width=True)
            board_col = get_col(hot_boards, 'board', strict=False)
            selected_boards = st.multiselect(
                "选择要检测的热门板块（可多选）",
                hot_boards[board_col].tolist() if board_col else [],
            )
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

    # ========= 分批分页逻辑 =========
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
            result_table.append({
                "代码": code,
                "信号": "、".join(signals) if signals else "无明显信号",
                "明细解释": "\n".join(explain),
                "AI点评": ai_result
            })
            if i < 6 and signals:  # 展示部分进度
                st.markdown(f"#### 【{code}】选股信号：{'、'.join(signals) if signals else '无明显信号'}")
                with st.expander("信号检测明细（点击展开）", expanded=False):
                    for line in explain:
                        st.write(line)
                if ai_result:
                    st.info(f"AI点评：{ai_result}")

        # 3. 展示/导出结果
        selected = [r for r in result_table if "无明显信号" not in r["信号"]]
        if selected:
            st.subheader("✅ 入选标的与信号（可全部导出，含AI点评）")
            df_sel = pd.DataFrame(selected)
            st.dataframe(df_sel[["代码","信号", "AI点评"]], use_container_width=True)
            st.download_button(
                "导出全部明细为Excel",
                data=pd.DataFrame(result_table).to_excel(index=False),
                file_name="AI选股明细.xlsx"
            )
        else:
            st.warning("暂无标的触发选股信号，可切换批次、调整策略或换池。")
    else:
        st.markdown("> 支持全A股、ETF、指数成分、热门池一键分批自动选股+AI趋势点评。")

# ======= tab2批量分析AI，保持你的原有风格 =======
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
