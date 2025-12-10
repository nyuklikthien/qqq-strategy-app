import yfinance as yf
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
from datetime import date, timedelta

# ======================================================
# 页面设定
# ======================================================
st.set_page_config(page_title="QQQ 策略观察器", layout="wide")
st.title("QQQ 策略观察器：定投管理 + 加仓管理（云端部署版）")


# ======================================================
# 抓取 QQQ 历史收盘价（带 cache）
# ======================================================
@st.cache_data
def load_clean_close(ticker: str):
    df = yf.download(ticker, start="1999-01-01")

    if df is None or df.empty:
        return pd.DataFrame()

    # MultiIndex 修复
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    cols = {c.lower(): c for c in df.columns}
    col = cols.get("adj close", cols.get("close", None))
    if col is None:
        return pd.DataFrame()

    s = pd.to_numeric(df[col], errors="coerce").dropna()

    out = pd.DataFrame({"Date": s.index, "Close": s.values})
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)


df_full = load_clean_close("QQQ")
if df_full.empty:
    st.error("无法取得 QQQ 数据，请稍后再试。")
    st.stop()

min_date = df_full["Date"].min().date()
max_date = df_full["Date"].max().date()


# ======================================================
# Sidebar：资产配置（定投管理）
# ======================================================
st.sidebar.header("资产配置（定投管理）")

monthly_income = st.sidebar.number_input(
    "每月可定投收入",
    min_value=0.0,
    value=3000.0,
    step=100.0,
)

freq_label = st.sidebar.radio(
    "定投时间（每月）",
    ["每月 1 次（1号）", "每月 2 次（1号 & 15号）"],
    index=1
)
dca_days = [1] if "1 次" in freq_label else [1, 15]

dca_ratio = st.sidebar.slider(
    "定投资金比例（每月收入 %）",
    0.0, 1.0, 0.4, 0.05
)

st.sidebar.caption("未投入部分累积到『加仓资金池』。")


# ======================================================
# Sidebar：加仓管理
# ======================================================
st.sidebar.header("加仓管理")

down_days = st.sidebar.number_input("连续下跌天数触发", 1, 30, 3)
base_dd_pct = st.sidebar.number_input("起始回撤 %（触发加仓）", 0.0, 90.0, 10.0)

base_add_ratio = st.sidebar.slider(
    "基础加仓比例（从加仓资金池拨出）",
    0.0, 1.0, 0.2, 0.05
)

step_dd_pct = st.sidebar.number_input("阶梯跌幅（每跌多少 %）", 0.1, 50.0, 3.0, 0.5)
step_add_ratio = st.sidebar.slider(
    "阶梯加仓增加 (%)",
    0.0, 0.5, 0.1, 0.05
)


# ======================================================
# Sidebar：回测区间
# ======================================================
st.sidebar.header("回测区间")

invest_start = st.sidebar.date_input(
    "投资开始日",
    value=max(min_date, date(2010, 1, 1)),
    min_value=min_date,
    max_value=max_date
)

invest_end = st.sidebar.date_input(
    "回测结束日",
    value=max_date,
    min_value=invest_start,
    max_value=max_date
)


# 截取回测区间数据
mask_bt = (
    (df_full["Date"].dt.date >= invest_start)
    & (df_full["Date"].dt.date <= invest_end)
)
df_bt = df_full.loc[mask_bt].reset_index(drop=True)

if len(df_bt) < 5:
    st.warning("区间内资料太少，请调整日期。")
    st.stop()


# ======================================================
# 策略模拟（已启用缓存）
# ======================================================
@st.cache_data
def simulate_strategy(
    df_bt,
    monthly_income, dca_days, dca_ratio,
    down_days, base_dd_pct,
    base_add_ratio, step_dd_pct, step_add_ratio,
    invest_start, invest_end
):
    dates = df_bt["Date"].dt.date.to_numpy()
    prices = df_bt["Close"].to_numpy()

    shares = 0.0
    addon_cash = 0.0
    total_invested = 0.0
    trades = []

    # 生成每月定投日
    from collections import defaultdict
    month_events = defaultdict(list)
    months = sorted({(d.year, d.month) for d in dates})

    for y, m in months:
        idxs = [i for i, d in enumerate(dates) if d.year == y and d.month == m]
        for dday in dca_days:
            sched = date(y, m, dday)
            if sched < invest_start or sched > invest_end:
                continue
            pick = None
            for i in idxs:
                if dates[i] >= sched:
                    pick = i
                    break
            if pick is not None:
                month_events[(y, m)].append(pick)

    # 回测主循环
    current_month = None
    consec_down = 0
    peak = None

    for i in range(len(dates)):
        d = dates[i]
        p = prices[i]
        ym = (d.year, d.month)

        # 新月份执行定投分配
        if ym != current_month:
            current_month = ym
            dca_total = monthly_income * dca_ratio
            addon_cash += monthly_income * (1 - dca_ratio)
            sched = month_events.get(ym, [])
            dca_per_trade = dca_total / len(sched) if sched else 0.0

        # 执行定投
        if i in month_events.get(ym, []) and invest_start <= d <= invest_end:
            if dca_per_trade > 0:
                buy = dca_per_trade
                add_s = buy / p
                shares += add_s
                addon_cash -= 0  # 不影响加仓池
                total_invested += buy
                trades.append({
                    "Type": "定投",
                    "Date": df_bt.loc[i, "Date"],
                    "Price": p,
                    "Amount": buy,
                    "Shares": add_s,
                    "Reason": f"定投 {d.day}号"
                })

        # 连跌与回撤状态
        if peak is None or p > peak:
            peak = p

        if i > 0 and p < prices[i - 1]:
            consec_down += 1
        else:
            consec_down = 0

        dd = (peak - p) / peak * 100 if peak > 0 else 0
        cond_down = consec_down >= down_days
        cond_dd = (base_dd_pct > 0) and (dd >= base_dd_pct)

        # 阶梯加仓
        if (cond_down or cond_dd) and addon_cash > 0 and base_add_ratio > 0:
            steps = max(0, int((dd - base_dd_pct) // step_dd_pct)) if dd > base_dd_pct else 0
            dyn_ratio = min(1.0, base_add_ratio + steps * step_add_ratio)
            buy_cash = addon_cash * dyn_ratio
            if buy_cash > 0:
                add_s = buy_cash / p
                shares += add_s
                addon_cash -= buy_cash
                total_invested += buy_cash
                trades.append({
                    "Type": "加仓",
                    "Date": df_bt.loc[i, "Date"],
                    "Price": p,
                    "Amount": buy_cash,
                    "Shares": add_s,
                    "Reason": f"回撤 {dd:.1f}%，阶梯 {steps}"
                })
                consec_down = 0
                peak = p

    # 期末
    final_value = shares * prices[-1] + addon_cash

    return trades, shares, addon_cash, total_invested, final_value


# 执行回测（超快，因为有缓存）
trades, shares, addon_cash, total_invested, final_value = simulate_strategy(
    df_bt,
    monthly_income, dca_days, dca_ratio,
    down_days, base_dd_pct,
    base_add_ratio, step_dd_pct, step_add_ratio,
    invest_start, invest_end
)


# ======================================================
# 图形显示选项（避免卡图）
# ======================================================
st.sidebar.header("图形显示选项")

marker_mode = st.sidebar.radio(
    "标记点显示：",
    ["不显示", "只显示定投", "只显示加仓", "全部"],
    index=0
)

trades_df = pd.DataFrame(trades)
if not trades_df.empty:
    trades_df["Date"] = pd.to_datetime(trades_df["Date"])
    dca_df = trades_df[trades_df["Type"] == "定投"]
    addon_df = trades_df[trades_df["Type"] == "加仓"]
else:
    dca_df = addon_df = pd.DataFrame(columns=["Date", "Price"])


# ======================================================
# TradingView 风格双图
# ======================================================
low = df_full["Close"].min() * 0.95
high = df_full["Close"].max() * 1.05

brush = alt.selection_interval(encodings=["x"])

base_line = alt.Chart(df_full).encode(
    x="Date:T",
    y=alt.Y("Close:Q", scale=alt.Scale(domain=[low, high]))
)

price_line = base_line.mark_line(color="#555")

# 定投点
dca_points = alt.Chart(dca_df).mark_circle(size=35, opacity=0.6, color="#2ecc71").encode(
    x="Date:T", y="Price:Q",
    tooltip=["Date:T", "Price:Q", "Amount:Q", "Shares:Q"]
)

dca_labels = alt.Chart(dca_df).mark_text(
    dy=-7, fontSize=8, color="#2ecc71"
).encode(
    x="Date:T", y="Price:Q", text=alt.Text("Price:Q", format=".2f")
)

# 加仓点
addon_points = alt.Chart(addon_df).mark_circle(size=35, opacity=0.7, color="#3498db").encode(
    x="Date:T", y="Price:Q",
    tooltip=["Date:T", "Price:Q", "Amount:Q", "Shares:Q"]
)

addon_labels = alt.Chart(addon_df).mark_text(
    dy=-7, fontSize=8, color="#3498db"
).encode(
    x="Date:T", y="Price:Q", text=alt.Text("Price:Q", format=".2f")
)

# 组合图层
layers = [price_line]

if marker_mode in ["只显示定投", "全部"]:
    layers += [dca_points, dca_labels]
if marker_mode in ["只显示加仓", "全部"]:
    layers += [addon_points, addon_labels]

main_chart = (
    alt.layer(*layers)
    .transform_filter(brush)
    .properties(height=320)
)

overview_chart = (
    alt.Chart(df_full)
    .mark_area(opacity=0.25)
    .encode(x="Date:T", y="Close:Q")
    .add_params(brush)
    .properties(height=80)
)

st.subheader("① QQQ 价格走势（可拖动区间）")
st.altair_chart(main_chart & overview_chart, use_container_width=True)


# ======================================================
# 回测结果
# ======================================================
st.subheader("② 回测结果（投资表现）")

c1, c2, c3, c4 = st.columns(4)
c1.metric("总投入", f"{total_invested:,.2f}")
c2.metric("期末总资产", f"{final_value:,.2f}")
c3.metric("持股数量", f"{shares:.4f}")
c4.metric("投资报酬率", f"{(final_value/total_invested-1)*100:.2f}%")

c5, c6 = st.columns(2)
c5.metric("加仓资金池余额", f"{addon_cash:,.2f}")
c6.metric("交易次数", len(trades))


# ======================================================
# 交易明细
# ======================================================
st.subheader("③ 交易明细")

if not trades_df.empty:
    df_show = trades_df.copy()
    df_show["Date"] = df_show["Date"].dt.strftime("%Y-%m-%d")
    df_show["Price"] = df_show["Price"].round(2)
    df_show["Amount"] = df_show["Amount"].round(2)
    df_show["Shares"] = df_show["Shares"].round(4)
    st.dataframe(df_show, hide_index=True, use_container_width=True)
else:
    st.info("本次回测没有任何交易。")
