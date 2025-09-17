import streamlit as st
import pandas as pd
import pandas_ta as ta
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import requests

st.set_page_config(page_title="A股批量智能技术分析 & AI趋势预测", layout="wide")
st.title("📈 A股批量AI自动选股 & 板块信号聚合解读")

# ============ 选股池工具函数 =============
@st.cache_data(show_spinner=False)
def get_all_a_codes():
    stock_df = ak.stock_info_a_code_name()
    return stock_df["code"].tolist()

@st.cache_data(show_spinner=False)
def get_all_etf_codes():
    etf_df = ak.fund_etf_category_sina(symbol="ETF基金")
    for col in ['symbol', '代码', 'fund']:
        if col in etf_df.columns:
            return etf_df[col].tolist()
    st.error(f"ETF数据字段异常: {etf_df.columns}")
    return []

@st.cache_data(show_spinner=False)
def get_index_codes(index_code):
    df = ak.index_stock_cons(index=index_code)
    return df["con_code"].tolist()

@st.cache_data(ttl=300, show_spinner=False)
def get_hot_concept_boards(topn=20):
    try:
        df = ak.stock_board_concept_name_ths()
        hot_df = df.sort_values("涨跌幅", ascending=False).head(topn)
        return hot_df[["板块名称", "涨跌幅"]]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_board_stocks(board_name):
    try:
        df = ak.stock_board_concept_cons_ths(symbol=board_name)
        return df["代码"].tolist()
    except Exception:
        return []

# ============ 板块归属映射 ============
@st.cache_data(ttl=3600)
def get_code_board_map():
    try:
        df = ak.stock_board_concept_cons_ths()
        code2board = defaultdict(list)
        for _, row in df.iterrows():
            code2board[row['代码']].append(row['板块名称'])
        return dict(code2board)
    except Exception:
        return {}
code2board = get_code_board_map()

# ============ 数据与指标函数 ============
def fetch_ak_data(code, start_date):
    # 支持多天回退，提升成功率
    for delta in range(0, 7):  # 尝试一周
        try:
            d = start_date - pd.Timedelta(days=delta)
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=d.strftime("%Y%m%d"), adjust="qfq")
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
            if "MACDh_12_26_9" in macd.columns:
                df["MACDh"] = macd["MACDh_12_26_9"]
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

# ============ DeepseeK AI调用 ============
def ai_trend_report(df, code, trend_days, deepseek_key):
    if not deepseek_key:
        return "未填写DeepseeK KEY，无法生成AI趋势预测。"
    use_df = df.tail(60)[["date", "open", "close", "high", "low", "volume"]]
    data_str = use_df.to_csv(index=False)
    prompt = f"""
你是一位A股专业量化分析师。以下是{code}最近60日的每日行情（日期,开盘,收盘,最高,最低,成交量），请根据技术走势、成交量变化，预测该股未来{trend_days}日的涨跌趋势，并判断是否存在启动信号、买卖机会，请以精炼中文输出一份点评。数据如下（csv格式）：
{data_str}
"""
    url = "https://api.deepseek.com/v1/chat/completions"  # 替换为你的DeepseeK实际API地址
    headers = {
        "Authorization": f"Bearer {deepseek_key.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位专业A股分析师。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.6,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception as ex:
        return f"DeepseeK分析调用失败：{ex}"

# ============ 分批分页主界面 ============
tab1, tab2 = st.tabs(["🪄 批量自动选股(分批+板块聚合)", "个股批量分析+AI点评"])

with tab1:
    st.subheader("全市场/ETF/指数/概念池自动选股，支持分批分析+板块聚合")
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
            st.dataframe(hot_boards)
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
    deepseek_key = st.text_input("如需AI批量趋势点评，请输入DeepseeK KEY", type="password", key="tab1_ai_key")
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

        for i, code in enumerate(codes_this_batch):
            df = result_dict.get(code, pd.DataFrame())
            if df.empty or len(df) < 25:
                continue
            df = calc_indicators(df)
            signals, explain = signal_with_explain(df)
            ai_result = ""
            if ai_batch and deepseek_key and signals:
                with st.spinner(f"AI分析{code}中..."):
                    ai_result = ai_trend_report(df, code, trend_days, deepseek_key)
                    time.sleep(1.2)  # 限速，防止被封
            # 板块列表
            board_list = code2board.get(code, ["未归属板块"])
            result_table.append({
                "代码": code,
                "板块": "、".join(board_list),
                "信号": "、".join(signals) if signals else "无明显信号",
                "明细解释": "\n".join(explain),
                "AI点评": ai_result
            })
            if i < 6 and signals:
                st.markdown(f"#### 【{code}】选股信号：{'、'.join(signals) if signals else '无明显信号'}")
                with st.expander("信号检测明细（点击展开）", expanded=False):
                    for line in explain:
                        st.write(line)
                if ai_result:
                    st.info(f"AI点评：{ai_result}")

              # ========== 板块聚合展示 ==========
        st.subheader("📊 本批选股信号板块分布与明细")
        board2stocks = defaultdict(list)
        for r in result_table:
            for board in r["板块"].split("、"):
                board2stocks[board].append(r)
        board_signal_summary = []
        for board, stock_list in board2stocks.items():
            cnt = len(stock_list)
            hit = sum("无明显信号" not in s["信号"] for s in stock_list)
            board_signal_summary.append({"板块": board, "本批股票数": cnt, "信号股数": hit})

        df_board = pd.DataFrame(board_signal_summary)
        if not df_board.empty and "信号股数" in df_board.columns:
            df_board = df_board.sort_values("信号股数", ascending=False)
            st.dataframe(df_board)
            selected_board = st.selectbox(
                "查看板块内信号明细", df_board["板块"].tolist() if not df_board.empty else ["无板块"])
            df_detail = pd.DataFrame(board2stocks.get(selected_board, []))
            if not df_detail.empty:
                st.dataframe(df_detail[["代码", "信号", "AI点评"]])
        else:
            st.info("本批没有检测到任何信号股，或板块分布为空。")

# ============= Tab2 =============
with tab2:
    st.subheader("自定义股票池批量分析+AI智能点评")
    deepseek_key = st.text_input("请输入你的DeepseeK API KEY（用于AI点评/趋势预测）", type="password", key="ai_key")
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
        st.plotly_chart(fig)
        if "MACD" in df.columns:
            fig2 = go.Figure()
            if "MACDh" in df.columns:
                fig2.add_trace(go.Bar(x=df["date"], y=df["MACDh"], name="MACD柱"))
            fig2.add_trace(go.Scatter(x=df["date"], y=df["MACD"], name="MACD线"))
            fig2.add_trace(go.Scatter(x=df["date"], y=df["MACDs"], name="信号线"))
            fig2.update_layout(title="MACD指标", height=200)
            st.plotly_chart(fig2)
        if "RSI_6" in df.columns:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df["date"], y=df["RSI_6"], name="RSI6"))
            fig3.update_layout(title="RSI指标", height=200, yaxis=dict(range=[0,100]))
            st.plotly_chart(fig3)

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
                    ai_report = ai_trend_report(df, code, trend_days, deepseek_key)
                    st.info(ai_report)
            st.divider()
    else:
        st.markdown("> 支持多只A股代码批量技术分析+AI自动点评（如需AI预测请填写DeepseeK KEY）")
