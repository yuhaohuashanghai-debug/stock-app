import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import datetime
import openai

def load_stock_data(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(symbol=symbol, start=start.strftime('%Y%m%d'), end=end.strftime('%Y%m%d'), adjust='qfq')
    df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'})
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    rs = roll_up / roll_down
    df['RSI'] = 100 - 100 / (1 + rs)
    df.dropna(inplace=True)
    return df


def detect_signals(df: pd.DataFrame) -> list:
    signals = []
    if df['MA5'].iloc[-2] < df['MA10'].iloc[-2] and df['MA5'].iloc[-1] > df['MA10'].iloc[-1]:
        signals.append('均线金叉')
    if df['DIF'].iloc[-2] < df['DEA'].iloc[-2] and df['DIF'].iloc[-1] > df['DEA'].iloc[-1]:
        signals.append('MACD金叉')
    if df['RSI'].iloc[-1] < 30:
        signals.append('RSI超卖')
    avg_vol = df['volume'].tail(5).mean()
    if df['volume'].iloc[-1] > 1.5 * avg_vol:
        signals.append('放量突破')
    if df['close'].iloc[-1] > df['close'].rolling(60).max().iloc[-2]:
        signals.append('新高')
    if df['close'].iloc[-1] < df['close'].rolling(60).min().iloc[-2]:
        signals.append('新低')
    return signals


st.set_page_config(page_title='选股与预测', layout='wide')

with st.sidebar:
    st.title('设置')
    openai_key = st.text_input('OpenAI API Key', type='password')
    if openai_key:
        openai.api_key = openai_key

st.header('板块排行')
industries, concepts = st.tabs(['行业', '概念'])
with industries:
    df_industry = ak.stock_board_industry_spot_em()
    st.dataframe(df_industry)
with concepts:
    df_concept = ak.stock_board_concept_spot_em()
    st.dataframe(df_concept)

st.header('成分股获取')
col1, col2 = st.columns(2)
with col1:
    board_type = st.selectbox('类别', ['指数', '板块'])
with col2:
    board_code = st.text_input('代码或名称', value='000300' if board_type == '指数' else '电力行业')
if st.button('获取成分股'):
    if board_type == '指数':
        comps = ak.index_stock_cons(symbol=board_code)
    else:
        comps = ak.stock_board_component_em(board=board_code)
    st.dataframe(comps)

st.header('K线图 + 均线 + MACD + RSI')
stock_code = st.text_input('股票代码', value='000001')
start_date = st.date_input('开始日期', value=datetime.date.today() - datetime.timedelta(days=365))
end_date = st.date_input('结束日期', value=datetime.date.today())
if st.button('绘制K线'):
    data = load_stock_data(stock_code, start_date, end_date)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=data['date'], open=data['open'], high=data['high'],
                                 low=data['low'], close=data['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['date'], y=data['MA5'], line=dict(color='orange'), name='MA5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data['date'], y=data['MA10'], line=dict(color='blue'), name='MA10'), row=1, col=1)
    fig.add_trace(go.Bar(x=data['date'], y=data['MACD'], name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['date'], y=data['DIF'], line=dict(color='red'), name='DIF'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['date'], y=data['DEA'], line=dict(color='green'), name='DEA'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data['date'], y=data['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
    fig.update_layout(xaxis3=dict(title='日期'), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

st.header('自动选股信号')
code_text = st.text_area('股票代码列表（逗号或换行分隔）', '000001\n000002')
if st.button('运行选股'):
    codes = [c.strip() for c in code_text.replace('\n', ',').split(',') if c.strip()]
    results = []
    for c in codes:
        try:
            df = load_stock_data(c, datetime.date.today() - datetime.timedelta(days=365), datetime.date.today())
            signals = detect_signals(df)
            results.append({'code': c, 'signals': ';'.join(signals)})
        except Exception as e:
            results.append({'code': c, 'signals': f'数据获取失败: {e}'})
    result_df = pd.DataFrame(results)
    st.dataframe(result_df)
    towrite = BytesIO()
    result_df.to_excel(towrite, index=False)
    towrite.seek(0)
    st.download_button('导出Excel', data=towrite, file_name='stock_signals.xlsx', mime='application/vnd.ms-excel')

st.header('AI趋势点评')
ai_code = st.text_input('AI点评股票代码')
if st.button('生成AI点评'):
    if not openai_key:
        st.warning('请在侧边栏输入 OpenAI API Key')
    else:
        try:
            df = load_stock_data(ai_code, datetime.date.today() - datetime.timedelta(days=120), datetime.date.today())
            recent = df[['date', 'close']].tail(30).to_json(orient='records')
            prompt = f"请根据以下近30日数据点评股票{ai_code}的趋势并给出简短建议：{recent}"
            response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role': 'user', 'content': prompt}])
            st.write(response['choices'][0]['message']['content'])
        except Exception as e:
            st.error(str(e))
