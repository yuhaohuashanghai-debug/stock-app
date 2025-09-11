# ChatGPT + 聚宽 股票分析工具

🚀 使用 Streamlit + ChatGPT 实现的智能股票分析工具，聚合技术面指标与 GPT 解读策略。

---

## 📦 功能特色

- 实时获取股票数据（聚宽接口）
- 支持 MACD / RSI 技术指标
- ChatGPT 自动策略建议分析
- 部署于 Streamlit Cloud 云平台

---

## 🔧 安装依赖

```bash
pip install -r requirements.txt
```

---

## 🚀 本地运行

```bash
streamlit run app.py
```

---

## ☁️ 云端部署（Streamlit Cloud）

1. 上传本项目至 GitHub
2. 创建 `.streamlit/secrets.toml`，内容如下：

```toml
[general]
openai_api_key = "你的OpenAI密钥"

[jq]
account = "你的聚宽账号"
password = "你的聚宽密码"
```

3. 部署时选择 `app.py` 作为主入口文件
