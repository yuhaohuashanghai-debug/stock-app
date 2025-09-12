        # --- 策略回测（form + session_state + 自动跳转锚点） ---
        st.subheader("📊 策略回测：MACD 金叉/死叉")

        with st.form("backtest_form"):
            col1, col2 = st.columns(2)
            with col1:
                lookback = st.number_input("回测天数 (lookback)", min_value=30, max_value=365, value=90, step=10)
            with col2:
                holding_days = st.number_input("持仓天数 (holding_days)", min_value=1, max_value=30, value=5, step=1)
            submitted = st.form_submit_button("运行回测")

        if submitted:
            results, trades = backtest_macd(df, lookback=lookback, holding_days=holding_days)
            st.session_state["backtest_results"] = results
            st.session_state["backtest_trades"] = trades
            st.session_state["lookback"] = lookback
            st.session_state["holding_days"] = holding_days
            # 设置 URL 参数，刷新后用于定位
            st.experimental_set_query_params(section="backtest")

        # 🚀 表单提交后刷新也能显示结果
        if "backtest_results" in st.session_state:
            # 定义一个锚点
            st.markdown("<a name='backtest'></a>", unsafe_allow_html=True)

            # 检查 query params 是否要求跳转
            params = st.experimental_get_query_params()
            if params.get("section", [""])[0] == "backtest":
                st.markdown(
                    """
                    <script>
                    var element = document.getElementsByName("backtest")[0];
                    if (element) {
                        element.scrollIntoView({behavior: "smooth", block: "start"});
                    }
                    </script>
                    """,
                    unsafe_allow_html=True
                )

            results = st.session_state["backtest_results"]
            trades = st.session_state["backtest_trades"]
            lookback = st.session_state["lookback"]
            holding_days = st.session_state["holding_days"]

            st.write(f"过去 {lookback} 天内：")
            st.write(f"- MACD 金叉次数: {results['金叉']['次数']}，{holding_days}日后上涨胜率: {results['金叉']['胜率']:.2%}")
            st.write(f"- MACD 死叉次数: {results['死叉']['次数']}，{holding_days}日后下跌胜率: {results['死叉']['胜率']:.2%}")

            if trades:
                st.write(f"最近几次交易回测记录 (持仓 {holding_days} 天)：")
                trade_df = pd.DataFrame(trades, columns=["信号", "日期", "买入价", "卖出价", "收益率"])
                trade_df["收益率"] = trade_df["收益率"].map(lambda x: f"{x:.2%}")
                st.dataframe(trade_df.tail(5))

                # 📥 下载按钮
                csv = trade_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="下载回测结果 CSV",
                    data=csv,
                    file_name=f"backtest_{code}.csv",
                    mime="text/csv"
                )
            else:
                st.info("⚠️ 最近没有检测到有效的 MACD 金叉/死叉信号，无法回测。")
