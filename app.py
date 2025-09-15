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
    code_candidates = ["symbol", "基金代码", "代码", "con_code", "成分券代码", "股票代码"]
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

# ====== 股票/ETF/指数接口 ======
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
    code_candidates = ["con_code", "成分券代码", "代码", "股票代码"]
    code_col = get_first_valid_column(df, code_candidates)
    return df[code_col].tolist()

# ====== 板块排行（实时版，含涨跌幅） ======
@st.cache_data(ttl=1800)
def get_hot_industry_boards(topn=20):
    df = ak.stock_board_industry_name_ths()
    show_columns(df, "行业板块")
    return sort_by_pct_chg(df, topn=topn)

@st.cache_data(ttl=1800)
def get_hot_concept_boards(topn=20):
    df = ak.stock_board_concept_name_ths()
    show_columns(df, "概念板块")
    return sort_by_pct_chg(df, topn=topn)

# ====== 板块成分股（行业 + 概念自动兼容） ======
@st.cache_data(ttl=300)
def get_board_stocks(board_name):
    try:
        df = ak.stock_board_concept_cons_ths(symbol=board_name)
    except Exception:
        try:
            df = ak.stock_board_industry_cons_ths(symbol=board_name)
        except Exception:
            return []
    return get_code_list(df) if not df.empty else []

# ====== K线与信号判别函数 ======
def fetch_ak_data(code, start_date):
    df = pd.DataFrame()
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start_date.strftime("%Y%m%d"), adjust="qfq")
        if not df.empty:
            df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                               "最高": "high", "最低": "low", "成交量": "volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
    except Exception:
        pass
    try:  # ETF兜底
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
    explain, signals = [], []
    latest = df.iloc[-1]
    pre = df.iloc[-2] if len(df) >= 2 else latest
    # 均线金叉
    if "SMA_5" in df.columns and "SMA_10" in df.columns:
        if pre["SMA_5"] < pre["SMA_10"] and latest["SMA_5"] > latest["SMA_10"]:
            signals.append("5日均线金叉10日均线")
            explain.append("【均线金叉】：今日5日均线上穿10日均线，多头信号。")
    # MACD金叉
    if "MACD" in df.columns and "MACDs" in df.columns:
        if pre["MACD"] < pre["MACDs"] and latest["MACD"] > latest["MACDs"]:
            signals.append("MACD金叉")
            explain.append("【MACD金叉】：今日MACD线上穿信号线，多头信号。")
    # RSI超卖
    if "RSI_6" in df.columns and latest["RSI_6"] < 30:
        signals.append("RSI6超卖反弹")
        explain.append("【RSI超卖反弹】：RSI跌破30，短期见底迹象。")
    return signals, explain

# ====== 三大 Tab 界面 ======
tab1, tab2, tab3 = st.tabs(["🔥 热门板块概念排行", "🪄 批量自动选股(分批)", "🤖 AI智能批量分析"])

# --- Tab1: 热门板块排行 ---
with tab1:
    st.subheader("今日热门行业/概念板块涨幅排行")
    col1, col2 = st.columns(2)
    with col1:
        industry_df = get_hot_industry_boards(topn=20)
        st.dataframe(industry_df, use_container_width=True)
        excel_bytes = dataframe_to_excel_bytes(industry_df)
        st.download_button("下载行业板块Excel", data=excel_bytes, file_name="行业板块.xlsx")
    with col2:
        concept_df = get_hot_concept_boards(topn=20)
        st.dataframe(concept_df, use_container_width=True)
        excel_bytes = dataframe_to_excel_bytes(concept_df)
        st.download_button("下载概念板块Excel", data=excel_bytes, file_name="概念板块.xlsx")

# --- Tab2: 分批自动选股 ---
with tab2:
    st.subheader("全市场/ETF/指数/概念池自动选股，支持分批分析")
    pool = st.selectbox("选择股票池", ["全A股","全ETF","沪深300","科创50","热门概念板块","自定义"])
    codes = []
    if pool == "全A股":
        codes = get_all_a_codes()
    elif pool == "全ETF":
        codes = get_all_etf_codes()
    elif pool == "沪深300":
        codes = get_index_codes_auto("000300")
    elif pool == "科创50":
        codes = get_index_codes_auto("000688")
    elif pool == "热门概念板块":
        hot = get_hot_concept_boards()
        st.dataframe(hot)
        selected = st.multiselect("选择热门板块", hot.iloc[:,0].tolist())
        for b in selected:
            codes += get_board_stocks(b)
    else:
        codes = [c.strip() for c in st.text_area("输入代码(逗号分隔)").split(",") if c.strip()]
    codes = list(set(codes))

    BATCH_SIZE = 200
    if "page" not in st.session_state: st.session_state["page"] = 0
    total_batches = (len(codes)-1)//BATCH_SIZE + 1 if codes else 1
    batch = st.session_state["page"]
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("上一批", disabled=(batch==0)): st.session_state["page"] -= 1
    with col2: st.write(f"第{batch+1}/{total_batches}批")
    with col3:
        if st.button("下一批", disabled=(batch+1>=total_batches)): st.session_state["page"] += 1
    codes_this = codes[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]

    start_date = st.date_input("起始日期", value=pd.to_datetime("2024-01-01"))
    if st.button("本批次自动选股"):
        result = []
        for code in codes_this:
            df = fetch_ak_data(code, start_date)
            if df.empty: continue
            df = calc_indicators(df)
            sigs, exp = signal_with_explain(df)
            result.append({"代码": code, "信号": "、".join(sigs) if sigs else "无", "解释": "\n".join(exp)})
        if result:
            df_sel = pd.DataFrame(result)
            st.dataframe(df_sel, use_container_width=True)
            excel_bytes = dataframe_to_excel_bytes(df_sel)
            st.download_button("导出选股结果", data=excel_bytes, file_name="AI选股结果.xlsx")

# --- Tab3: AI批量分析 ---
with tab3:
    st.subheader("AI智能点评（需OpenAI API Key）")
    key = st.text_input("输入OpenAI API Key", type="password")
    codes = st.text_input("输入代码(如 000977,600519,588170)", "000977,518880").split(",")
    start_date = st.date_input("起始日期", value=(datetime.now()-pd.Timedelta(days=180)).date())
    days = st.selectbox("预测天数", [1,3,5,7], index=1)
    if st.button("开始AI分析"):
        for code in codes:
            code = code.strip()
            if not code: continue
            st.markdown(f"### 【{code}】分析")
            df = fetch_ak_data(code, start_date)
            if df.empty:
                st.warning(f"{code} 无数据")
                continue
            df = calc_indicators(df)
            st.dataframe(df.tail(10))
            fig = go.Figure(data=[go.Candlestick(x=df["date"],open=df["open"],high=df["high"],
                                                 low=df["low"],close=df["close"])])
            st.plotly_chart(fig, use_container_width=True)
            if key:
                try:
                    import openai
                    client = openai.OpenAI(api_key=key)
                    data_str = df.tail(60)[["date","open","close","high","low","volume"]].to_csv(index=False)
                    prompt = f"请基于以下数据预测 {code} 未来{days}日趋势：\n{data_str}"
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role":"system","content":"你是一位专业A股分析师"},
                                  {"role":"user","content":prompt}],
                        max_tokens=400
                    )
                    st.info(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI调用失败: {e}")

st.info("版本已修复接口兼容问题，可直接部署运行。")
