import streamlit as st
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_by_gpt(stock_code, last_row):
    prompt = f"请分析股票 {stock_code} 的最新技术指标数据：{last_row}，并给出策略建议"
    res = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

st.title("📊 AkShare + ChatGPT 技术面股票分析")
stock_code = st.text_input("输入股票代码", "000001.SZ")
last_row = {"MACD": 0.02, "RSI": 58.1}  # 示例数据

if st.button("生成策略建议"):
    suggestion = explain_by_gpt(stock_code, last_row)
    st.success(suggestion)
