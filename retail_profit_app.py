import streamlit as st
import numpy as np
import pandas as pd

# ===================== 工具函数 =====================
def make_curve(value_or_list, default_val=0.0):
    """统一输入为24小时曲线"""
    if value_or_list is None:
        return np.array([default_val] * 24, dtype=float)
    if isinstance(value_or_list, (int, float)):
        return np.array([float(value_or_list)] * 24, dtype=float)
    if isinstance(value_or_list, (list, tuple, np.ndarray)) and len(value_or_list) == 24:
        return np.array(value_or_list, dtype=float)
    return np.array([default_val] * 24, dtype=float)


# ===================== 参数检查 =====================
def enforce_rules(params: dict):
    """根据安徽电力零售合同注释3-10进行校验"""
    warnings = []

    # 3. 套餐比例总和 ≤ 100%
    total_ratio = params["k1_ratio"] + params["k2_ratio"] + params["k3_ratio"]
    if total_ratio > 1:
        warnings.append(f"套餐比例总和 {total_ratio:.2f} 超过 100%，系统已自动等比例缩放。")
        scale = 1 / total_ratio
        for k in ["k1_ratio", "k2_ratio", "k3_ratio"]:
            params[k] *= scale

    # 6/7. 套餐三分成比例 ≥ 50%
    if params["k3_ratio"] > 0 and params["k3_share_ratio"] < 0.5:
        warnings.append("套餐三分成比例低于 50%，系统已自动调整为 50%。")
        params["k3_share_ratio"] = 0.5

    return params, warnings


# ===================== 收益计算 =====================
def calc_profit_detailed(
    total_power_mwh, wholesale_price,
    k1_curve, k1_ratio,
    market_avg_price, k2_float_percent, k2_ratio,
    k3_input_curve, k3_is_factor, k3_share_ratio, k3_ratio,
    green_ratio, green_fix_price,
):
    """
    收益测算主函数
    所有价格单位：元/kWh；电量单位：MWh / kWh
    """
    total_power_kwh = total_power_mwh * 1000.0
    if total_power_kwh <= 0:
        return {}

    w_curve = make_curve(wholesale_price)

    # ---- K1 固定价 ----
    k1_curve = make_curve(k1_curve)
    profit_k1 = (k1_curve - w_curve).mean() * k1_ratio * total_power_kwh

    # ---- K2 市场均价 + 浮动 ----
    p2_price = market_avg_price * (1 + k2_float_percent)
    profit_k2 = (p2_price - wholesale_price) * k2_ratio * total_power_kwh

    # ---- K3 价差分成 ----
    market_curve = make_curve(market_avg_price)
    if k3_is_factor:
        base_curve = market_curve * make_curve(k3_input_curve, 1.0)
    else:
        base_curve = make_curve(k3_input_curve, market_avg_price)
    # P3T = P基T − (P基T − P售均T) × K分成%
    p3_curve = base_curve - (base_curve - market_curve) * k3_share_ratio
    profit_k3 = (p3_curve - w_curve).mean() * k3_ratio * total_power_kwh

    # ---- 综合价 & 绿电收益 ----
    total_ratio = max(k1_ratio + k2_ratio + k3_ratio, 1e-6)
    blended_price = (
        k1_curve.mean() * k1_ratio +
        p2_price * k2_ratio +
        p3_curve.mean() * k3_ratio
    ) / total_ratio

    # 绿电：只按固定环境权益单价计费（不再有百分比溢价）
    green_profit = total_power_kwh * green_ratio * green_fix_price

    total_profit = profit_k1 + profit_k2 + profit_k3 + green_profit
    unit_profit = total_profit / total_power_kwh

    return {
        "总收益(元)": round(total_profit, 2),
        "单位收益(元/kWh)": round(unit_profit, 4),
        "K1收益(元)": round(profit_k1, 2),
        "K2收益(元)": round(profit_k2, 2),
        "K3收益(元)": round(profit_k3, 2),
        "绿电收益(元)": round(green_profit, 2),
        "平均零售价(不含绿电)": round(blended_price, 4),
        "K1平均价(元/kWh)": round(k1_curve.mean(), 4),
        "K2结算价(元/kWh)": round(p2_price, 4),
        "K3平均价(元/kWh)": round(p3_curve.mean(), 4),
    }


# ===================== 敏感性分析 =====================
def sensitivity_analysis(market_avg_price, **kwargs):
    """市场均价±30%收益曲线"""
    changes = np.arange(-0.3, 0.31, 0.05)
    data = []
    for c in changes:
        new_market = market_avg_price * (1 + c)
        res = calc_profit_detailed(market_avg_price=new_market, **kwargs)
        if res:
            data.append([round(c * 100, 0), res["单位收益(元/kWh)"]])
    return pd.DataFrame(data, columns=["市场均价变动(%)", "单位收益(元/kWh)"])


# ===================== 页面 =====================
st.set_page_config(page_title="安徽电力零售收益模拟器", layout="wide")
st.title("⚡ 安徽电力零售收益模拟器（2026合同版）")

# ---- 基础参数 ----
st.subheader("一、基础参数")
b1, b2, b3 = st.columns(3)
total_power_mwh = b1.number_input("售电总电量 (MWh)", 0.0, 100000.0, 5000.0, step=100.0)
wholesale_price = b1.number_input("批发购电成本 (元/kWh)", 0.0000, 2.0000, 0.6800, step=0.0001, format="%.4f")

market_avg_price = b2.number_input("市场均价 P_market (元/kWh)", 0.0000, 2.0000, 0.7500, step=0.0001, format="%.4f")
p_settle_last_year = b2.number_input("上一年度批发侧结算均价 P_settle_2025 (元/kWh)",
                                     0.0000, 2.0000, 0.7300, step=0.0001, format="%.4f")
p_green_avg = b2.number_input("绿色电力批发侧均值 P_green_avg (元/kWh)",
                              0.0000, 2.0000, 0.0300, step=0.0001, format="%.4f")

k1_ratio = b3.slider("套餐一 K1 比例", 0.0, 1.0, 0.4, 0.05)
k2_ratio = b3.slider("套餐二 K2 比例", 0.0, 1.0, 0.4, 0.05)
k3_ratio = b3.slider("套餐三 K3 比例", 0.0, 1.0, 0.2, 0.05)

st.divider()

# ---- 套餐一 ----
st.subheader("二、套餐一：固定电价套餐（K1）")
k1_mode = st.radio("输入方式", ["统一电价", "24时点曲线"], horizontal=True)
if k1_mode == "统一电价":
    k1_flat = st.number_input("统一固定价 P1 (元/kWh)", 0.0000, 2.0000, 0.7000, step=0.0001, format="%.4f")
    k1_curve = make_curve(k1_flat)
else:
    st.markdown("请输入 24 个时点固定价（元/kWh）：")
    k1_vals = [
        st.number_input(f"{i}时", 0.0000, 2.0000, 0.7000,
                        step=0.0001, format='%.4f', key=f"k1_{i}")
        for i in range(1, 25)
    ]
    k1_curve = make_curve(k1_vals, 0.0)

st.divider()

# ---- 套餐二 ----
st.subheader("三、套餐二：市场均价 + 浮动价格套餐（K2）")
p_float_input = st.number_input(
    "请输入浮动价格 P浮动T (元/kWh)",
    -0.5000, 0.5000, 0.0000,
    step=0.0001, format="%.4f",
    help="填写相对于市场均价的加减浮动价（正值上浮，负值下浮）",
)
p2_price = market_avg_price + p_float_input
st.write(f"当前结算价 P2T = {p2_price:.4f} 元/kWh")

diff_ratio = p_float_input / p_settle_last_year * 100
if abs(diff_ratio) > 3:
    st.warning(
        f"⚠️ 相对上一年度批发侧结算均价 {p_settle_last_year:.4f} 元/kWh，"
        f"变动 {diff_ratio:.2f}%，超过 ±3%，按合同需法人代表确认。"
    )
else:
    st.success(f"✅ 相对上一年度批发侧结算均价变动 {diff_ratio:.2f}%，符合 ±3% 范围。")

k2_float_percent = p_float_input / market_avg_price
st.caption(f"对应浮动比例约为：{k2_float_percent * 100:.2f}%")

st.divider()

# ---- 套餐三 ----
st.subheader("四、套餐三：价差分成套餐（K3）")
k3_mode = st.radio(
    "请选择基价形式",
    ["统一浮动系数（P基T = P售均 × K浮动）", "统一基准价（固定值）", "24时点基准价（曲线）"],
    horizontal=True,
)
k3_is_factor = (k3_mode == "统一浮动系数（P基T = P售均 × K浮动）")

if k3_is_factor:
    k3_float = st.number_input(
        "统一浮动系数 K浮动（例如 1.05 = 上浮5%）",
        0.8000, 2.0000, 1.0500,
        step=0.0001, format="%.4f",
    )
    k3_input_curve = make_curve(k3_float)
    p_base = market_avg_price * k3_float
    st.write(f"当前基准价 P基T = {p_base:.4f} 元/kWh （市场均价 {market_avg_price:.4f}）")

    if k3_float > 1.05:
        st.warning(
            f"⚠️ 浮动系数 {k3_float:.4f} 超过 1.05（上浮5%），按合同需法人授权确认。"
        )
    else:
        st.success(f"✅ 浮动系数 {k3_float:.4f} 在允许范围内。")
else:
    if k3_mode == "统一基准价（固定值）":
        k3_base_price = st.number_input(
            "统一基准价 P基 (元/kWh)",
            0.0000, 2.5000, market_avg_price,
            step=0.0001, format="%.4f",
        )
        k3_input_curve = make_curve(k3_base_price)
    else:
        st.markdown("请输入 24 个时点固定基准价 P基T（单位：元/kWh）：")
        k3_vals = [
            st.number_input(
                f"{i}时", 0.0000, 2.5000, market_avg_price,
                step=0.0001, format='%.4f', key=f'k3_base_{i}'
            )
            for i in range(1, 25)
        ]
        k3_input_curve = make_curve(k3_vals, market_avg_price)

k3_share_ratio = st.slider("价差分成比例 K分成（50%~100%）", 0.5, 1.0, 0.8, 0.05)

st.divider()

# ---- 绿电：只保留“固定环境权益价” ----
st.subheader("五、套餐五：绿色电力环境权益（K5）")
g1, g2 = st.columns(2)
green_ratio = g1.slider("绿电比例", 0.0, 1.0, 0.2, 0.05)
green_fix_price = g2.number_input(
    "环境权益价 P_green_fix (元/kWh)",
    0.0000, 2.0000, 0.0200,
    step=0.0001, format="%.4f",
)

if p_green_avg > 0:
    ratio_to_avg = green_fix_price / p_green_avg * 100
    st.write(f"当前环境权益价为批发侧均值的 **{ratio_to_avg:.2f}%**。")
    if ratio_to_avg > 110:
        st.warning(
            f"⚠️ 当前环境权益价 {green_fix_price:.4f} 元/kWh 高于批发均值 "
            f"{p_green_avg:.4f} 的 110%，需市场风险提示。"
        )
    else:
        st.success("✅ 环境权益价在允许范围内（≤110%）。")
else:
    st.info("请输入绿色电力批发侧均值以进行风险比例判断。")

# ---- 校验与计算 ----
params = {
    "k1_ratio": k1_ratio,
    "k2_ratio": k2_ratio,
    "k3_ratio": k3_ratio,
    "k2_float_percent": k2_float_percent,
    "market_avg_price": market_avg_price,
    "k3_share_ratio": k3_share_ratio,
}
params, warnings = enforce_rules(params)

results = {}
if total_power_mwh > 0:
    results = calc_profit_detailed(
        total_power_mwh, wholesale_price,
        k1_curve, params["k1_ratio"],
        params["market_avg_price"], params["k2_float_percent"], params["k2_ratio"],
        k3_input_curve, k3_is_factor, params["k3_share_ratio"], params["k3_ratio"],
        green_ratio, green_fix_price,
    )

# ---- 输出 ----
st.subheader("六、参数与结果")
if warnings:
    st.warning("⚠️ 存在以下提示：")
    for w in warnings:
        st.write("-", w)
if results:
    st.json(results)

st.subheader("七、市场均价 ±30% 敏感性分析")
if results:
    sens_df = sensitivity_analysis(
        market_avg_price=params["market_avg_price"],
        total_power_mwh=total_power_mwh,
        wholesale_price=wholesale_price,
        k1_curve=k1_curve,
        k1_ratio=params["k1_ratio"],
        k2_float_percent=params["k2_float_percent"],
        k2_ratio=params["k2_ratio"],
        k3_input_curve=k3_input_curve,
        k3_is_factor=k3_is_factor,
        k3_share_ratio=params["k3_share_ratio"],
        k3_ratio=params["k3_ratio"],
        green_ratio=green_ratio,
        green_fix_price=green_fix_price,
    )
    st.line_chart(sens_df.set_index("市场均价变动(%)"))

