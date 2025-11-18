import streamlit as st
import numpy as np
import pandas as pd

# =====================================================================
# 固定数据：安徽 2025 火电基准曲线 —— 先按占比再用于中长期分配
# =====================================================================
FIRE_CURVE = np.array([
    77.683, 90.015, 90.380, 88.920, 91.016, 90.825,
    81.438, 70.063, 65.534, 48.805, 51.304, 51.038,
    51.064, 50.161, 49.342, 64.623, 83.712, 82.966,
    78.767, 72.749, 67.500, 69.336, 68.299, 72.468
], dtype=float)
FIRE_CURVE = FIRE_CURVE / FIRE_CURVE.sum()   # → 转成占比，用来分配 85% 中长期电量

# =====================================================================
# 安徽尖峰平谷时段定义
# =====================================================================
TOU_TABLE = {
    1:  ["谷","谷","谷","谷","谷","谷","谷","谷","平","平","平","平","平","平","平","峰","峰","峰","峰","峰","峰","峰","峰","谷"],
    2:  ["谷","谷","谷","谷","谷","谷","峰","峰","平","平","平","平","谷","谷","平","平","平","峰","峰","峰","峰","峰","峰","谷"],
    3:  ["谷","谷","谷","谷","谷","谷","峰","峰","平","平","平","平","谷","谷","平","平","平","峰","峰","峰","峰","峰","峰","谷"],
    4:  ["谷","谷","谷","谷","谷","谷","峰","峰","平","平","平","平","谷","谷","平","平","平","峰","峰","峰","峰","峰","峰","谷"],
    5:  ["谷","谷","谷","谷","谷","谷","峰","峰","平","平","平","平","谷","谷","平","平","平","峰","峰","峰","峰","峰","峰","谷"],
    6:  ["谷","谷","谷","谷","谷","谷","峰","峰","平","平","平","平","谷","谷","平","平","平","峰","峰","峰","峰","峰","峰","谷"],
    7:  ["谷","谷","谷","谷","谷","谷","谷","谷","谷","平","平","平","平","平","平","平","峰","峰","峰","峰","峰","峰","峰","峰"],
    8:  ["谷","谷","谷","谷","谷","谷","谷","谷","谷","平","平","平","平","平","平","平","峰","峰","峰","峰","峰","峰","峰","峰"],
    9:  ["谷","谷","谷","谷","谷","谷","谷","谷","谷","平","平","平","平","平","平","平","峰","峰","峰","峰","峰","峰","峰","峰"],
    10: ["谷","谷","谷","谷","谷","谷","峰","峰","平","平","平","平","谷","谷","平","平","平","峰","峰","峰","峰","峰","峰","谷"],
    11: ["谷","谷","谷","谷","谷","谷","峰","峰","平","平","平","平","谷","谷","平","平","平","峰","峰","峰","峰","峰","峰","谷"],
    12: ["谷","谷","谷","谷","谷","谷","谷","谷","平","平","平","平","平","平","平","峰","峰","峰","尖","尖","尖","峰","峰","谷"],
}
# =====================================================================
# 通用工具函数
# =====================================================================
def make_curve(value_or_list, default_val=0.0):
    """
    把单值或列表统一转换成 24 点曲线（numpy array）
    """
    if value_or_list is None:
        return np.array([default_val] * 24, dtype=float)
    if isinstance(value_or_list, (int, float)):
        return np.array([float(value_or_list)] * 24, dtype=float)
    if isinstance(value_or_list, (list, tuple, np.ndarray)) and len(value_or_list) == 24:
        return np.array(value_or_list, dtype=float)
    return np.array([default_val] * 24, dtype=float)


def split_load_by_tou(month,尖,峰,平,谷):
    """
    根据月份，把尖/峰/平/谷月电量平均拆分到 24 个小时
    返回：长度 24 的用户用电曲线（MWh）
    """
    table = TOU_TABLE[month]
    count_尖 = table.count("尖") if "尖" in table else 0
    count_峰 = table.count("峰")
    count_平 = table.count("平")
    count_谷 = table.count("谷")

    arr = []
    for t in table:
        if t == "尖" and count_尖 > 0:
            arr.append(尖 / count_尖 if count_尖 > 0 else 0.0)
        elif t == "峰":
            arr.append(峰 / count_峰 if count_峰 > 0 else 0.0)
        elif t == "平":
            arr.append(平 / count_平 if count_平 > 0 else 0.0)
        elif t == "谷":
            arr.append(谷 / count_谷 if count_谷 > 0 else 0.0)
        else:
            arr.append(0.0)
    return np.array(arr, dtype=float)


def make_contract_curve(user_curve, fire_curve):
    """
    中长期电量 = 总用电量 * 85% * 火电曲线占比
    """
    total_user = user_curve.sum()
    return total_user * 0.85 * fire_curve


def calc_cost(user_curve, contract_curve, long_curve_price, spot_curve):
    """
    购电成本 = 中长期成本 + 偏差成本（现货）
    long_curve_price: 24 点中长期结算单价曲线（元/kWh）
    spot_curve: 24 点现货电价曲线（元/kWh）
    """
    dev = user_curve - contract_curve   # 偏差电量（正：在现货买；负：在现货卖）
    long_cost = np.sum(contract_curve * long_curve_price)
    spot_cost = np.sum(dev * spot_curve)
    total_cost = long_cost + spot_cost
    avg_cost = total_cost / max(user_curve.sum(), 1e-9)
    return total_cost, avg_cost, dev


# =====================================================================
# Streamlit 页面配置
# =====================================================================
st.set_page_config(page_title="安徽电力零售收益模拟器", layout="wide")
st.title("⚡ 安徽电力零售收益模拟器（升级成本精算版）")

# =====================================================================
# 一、成本精算模块
# =====================================================================
st.header("一、成本精算模块（85% 中长期 + 15% 现货偏差）")

# ---- 月份 & 中长期基准价 ----
row1_col1, row1_col2 = st.columns([1, 1])
with row1_col1:
    month = st.selectbox("选择月份（用于尖峰平谷拆分）", list(range(1, 13)), index=0)
with row1_col2:
    P_long = st.number_input(
        "中长期基准价 P_long (元/kWh)",
        0.0000, 3.0000, 0.6500, step=0.0001, format="%.4f"
    )

# ---- 尖峰平谷 4 个电量 ----
c1, c2, c3, c4 = st.columns(4)
with c1:
   尖 = st.number_input("尖电量 (MWh)", 0.0, 1e9, 0.0)
with c2:
   峰 = st.number_input("峰电量 (MWh)", 0.0, 1e9, 0.0)
with c3:
   平 = st.number_input("平电量 (MWh)", 0.0, 1e9, 0.0)
with c4:
   谷 = st.number_input("谷电量 (MWh)", 0.0, 1e9, 0.0)

# ---- 生成用户 24 点用电曲线 ----
user_curve = split_load_by_tou(month, 尖, 峰, 平, 谷)

# 中长期单价曲线：以火电曲线相对平均值的形状拉伸到 P_long
rel = FIRE_CURVE / FIRE_CURVE.mean()
long_curve_price = rel * P_long

# ---- 日前现货价格输入----
st.subheader("二、日前现货价格曲线（元/kWh）")

use_custom_spot = st.checkbox("手动输入 24 点现货价格（不勾选则使用统一默认值）", value=False)

if use_custom_spot:
    spot_vals = [0.35] * 24
    rows = 4
    cols_per_row = 6
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            i = r * cols_per_row + c
            with cols[c]:
                spot_vals[i] = st.number_input(
                    f"{i}:00",
                    0.0000, 5.0000, float(spot_vals[i]),
                    step=0.0001, format="%.4f", key=f"spot_{i}"
                )
    spot_curve = np.array(spot_vals, dtype=float)
else:
    # 简单双曲线可以后续再调整，这里先统一值
    spot_curve = np.array([0.3500] * 24, dtype=float)

# ---- 成本计算 ----
contract_curve = make_contract_curve(user_curve, FIRE_CURVE)
total_cost, avg_cost, dev_curve = calc_cost(
    user_curve, contract_curve, long_curve_price, spot_curve
)

st.success(f"📌 当前按输入测算的 **平均购电成本 = {avg_cost:.4f} 元/kWh**")

# ---- 曲线展示 ----
st.subheader("三、用户用电 / 中长期 / 偏差曲线（MWh）")

df_chart = pd.DataFrame({
    "用户用电(MWh)": user_curve,
    "中长期合同(MWh)": contract_curve,
    "偏差电量(MWh)": dev_curve
}, index=[f"{i}:00" for i in range(24)])

st.line_chart(df_chart)

st.divider()

# =====================================================================
# 二、收益测算模块（K1 / K2 / K3 / 绿电）
# =====================================================================
st.header("二、收益测算模块（K1/K2/K3/绿电）")
st.caption("说明：本模块中“批发购电成本”已自动使用上方测算得到的 avg_cost。")

# ---- 基础参数（收益模块）----
b1, b2, b3 = st.columns(3)

with b1:
    total_power_mwh = st.number_input(
        "售电总电量 (MWh)",
        0.0, 1e9,
        float(user_curve.sum()),  # 默认用成本模块的总用电量
        step=100.0
    )
    # 批发成本 = 上面成本模块测出来的平均成本
    wholesale_price = avg_cost

with b2:
    market_avg_price = st.number_input(
        "市场均价 P_market (元/kWh)",
        0.0000, 2.0000, 0.7500, step=0.0001, format="%.4f"
    )
    p_settle_last_year = st.number_input(
        "上一年度批发侧结算均价 (元/kWh)",
        0.0000, 2.0000, 0.7300, step=0.0001, format="%.4f"
    )
    p_green_avg = st.number_input(
        "绿色电力批发均值 (元/kWh)",
        0.0000, 2.0000, 0.0300, step=0.0001, format="%.4f"
    )

with b3:
    k1_ratio = st.slider("K1 比例", 0.0, 1.0, 0.4, 0.05)
    k2_ratio = st.slider("K2 比例", 0.0, 1.0, 0.4, 0.05)
    k3_ratio = st.slider("K3 比例", 0.0, 1.0, 0.2, 0.05)

# =====================================================================
# 收益模块内部工具函数（沿用原来的逻辑）
# =====================================================================
def enforce_rules_profit(params: dict):
    """套餐比例 & K3 分成比例等合规校验"""
    warnings = []
    total_ratio = params["k1_ratio"] + params["k2_ratio"] + params["k3_ratio"]
    if total_ratio > 1:
        warnings.append(f"套餐比例总和 {total_ratio:.2f} 超过 100%，系统已自动等比例缩放。")
        scale = 1 / total_ratio
        for k in ["k1_ratio", "k2_ratio", "k3_ratio"]:
            params[k] *= scale
    if params["k3_ratio"] > 0 and params["k3_share_ratio"] < 0.5:
        warnings.append("套餐三分成比例低于 50%，系统已自动调整为 50%。")
        params["k3_share_ratio"] = 0.5
    return params, warnings


def calc_profit_detailed(
    total_power_mwh, wholesale_price,
    k1_curve, k1_ratio,
    market_avg_price, k2_float_percent, k2_ratio,
    k3_input_curve, k3_is_factor, k3_share_ratio, k3_ratio,
    green_ratio, green_fix_price,
):
    """
    套餐收益测算
    """
    total_power_kwh = total_power_mwh * 1000.0
    if total_power_kwh <= 0:
        return {}

    w_curve = make_curve(wholesale_price)  # 批发成本曲线（统一价）
    k1_curve = make_curve(k1_curve)
    profit_k1 = (k1_curve - w_curve).mean() * k1_ratio * total_power_kwh

    # K2：市场均价 + 浮动价
    p2_price = market_avg_price * (1 + k2_float_percent)
    profit_k2 = (p2_price - wholesale_price) * k2_ratio * total_power_kwh

    # K3：价差分成
    market_curve = make_curve(market_avg_price)
    if k3_is_factor:
        base_curve = market_curve * make_curve(k3_input_curve, 1.0)
    else:
        base_curve = make_curve(k3_input_curve, market_avg_price)

    # P3T = P基T − (P基T − P售均T) × K分成
    p3_curve = base_curve - (base_curve - market_curve) * k3_share_ratio
    profit_k3 = (p3_curve - w_curve).mean() * k3_ratio * total_power_kwh

    # 加权平均零售价（不含绿电）
    blend_ratio = max(k1_ratio + k2_ratio + k3_ratio, 1e-6)
    blended_price = (
        k1_curve.mean() * k1_ratio +
        p2_price * k2_ratio +
        p3_curve.mean() * k3_ratio
    ) / blend_ratio

    # 绿电附加（只用固定价，不再有百分比溢价）
    green_profit = total_power_kwh * green_ratio * green_fix_price

    total_profit = profit_k1 + profit_k2 + profit_k3 + green_profit
    unit_profit = total_profit / max(total_power_kwh, 1e-9)

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


# =====================================================================
# K1 固定价套餐
# =====================================================================
st.subheader("K1 固定价套餐")

k1_mode = st.radio("K1 输入方式", ["统一固定价", "24 时点曲线"], horizontal=True)

if k1_mode == "统一固定价":
    k1_flat = st.number_input(
        "固定电价 P1 (元/kWh)",
        0.0000, 2.0000, 0.7000, step=0.0001, format="%.4f"
    )
    k1_curve = make_curve(k1_flat)
else:
    st.markdown("请输入 24 点 K1 电价曲线（元/kWh）：")
    k1_vals = [0.70] * 24
    rows = 4
    cols_per_row = 6
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            i = r * cols_per_row + c
            with cols[c]:
                k1_vals[i] = st.number_input(
                    f"K1 {i}:00",
                    0.0000, 2.0000, float(k1_vals[i]),
                    step=0.0001, format="%.4f", key=f"k1_{i}"
                )
    k1_curve = make_curve(k1_vals)

# =====================================================================
# K2 市场均价 + 浮动套餐
# =====================================================================
st.subheader("K2 市场加浮动套餐")

p_float_input = st.number_input(
    "K2 浮动价 ΔP (元/kWh，相对市场均价加减)",
    -1.0000, 1.0000, 0.0000,
    step=0.0001, format="%.4f"
)
p2_price_preview = market_avg_price + p_float_input
k2_float_percent = p_float_input / market_avg_price if market_avg_price > 0 else 0.0
st.caption(f"当前 K2 结算电价约为：{p2_price_preview:.4f} 元/kWh，对应浮动比例 {k2_float_percent*100:.2f}%")

diff_ratio = p_float_input / p_settle_last_year * 100 if p_settle_last_year > 0 else 0.0
if abs(diff_ratio) > 3:
    st.warning(
        f"⚠️ 相对上一年度批发侧结算均价 {p_settle_last_year:.4f} 元/kWh，"
        f"浮动价对应变动 {diff_ratio:.2f}% ，超过 ±3%，按合同需法人代表确认。"
    )

# =====================================================================
# K3 价差分成套餐
# =====================================================================
st.subheader("K3 价差分成套餐")

k3_mode = st.radio(
    "K3 基准价形式",
    ["浮动系数（P基T = P售均 × K浮动）", "统一基准价", "24 点基准价曲线"],
    horizontal=True,
)

k3_is_factor = (k3_mode.startswith("浮动系数"))

if k3_is_factor:
    k3_float = st.number_input(
        "统一浮动系数 K浮动（例如 1.05 = 上浮 5%）",
        0.5000, 2.0000, 1.0500,
        step=0.0001, format="%.4f"
    )
    k3_input_curve = make_curve(k3_float)
    p_base = market_avg_price * k3_float
    st.caption(f"当前 K3 基准价 P基T ≈ {p_base:.4f} 元/kWh")
    if k3_float > 1.05:
        st.warning("⚠️ K3 浮动系数超过 1.05（上浮 5%），按合同需法人授权确认。")
else:
    if k3_mode == "统一基准价":
        base = st.number_input(
            "统一基准价 P基 (元/kWh)",
            0.0000, 3.0000, market_avg_price,
            step=0.0001, format="%.4f"
        )
        k3_input_curve = make_curve(base)
    else:
        st.markdown("请输入 24 点 K3 基准价曲线 P基T（元/kWh）：")
        k3_vals = [market_avg_price] * 24
        rows = 4
        cols_per_row = 6
        for r in range(rows):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                i = r * cols_per_row + c
                with cols[c]:
                    k3_vals[i] = st.number_input(
                        f"K3 {i}:00",
                        0.0000, 3.0000, float(k3_vals[i]),
                        step=0.0001, format="%.4f", key=f"k3_{i}"
                    )
        k3_input_curve = make_curve(k3_vals)

k3_share_ratio = st.slider("K3 价差分成比例（50%~100%）", 0.5, 1.0, 0.8, 0.05)

# =====================================================================
# 绿电套餐
# =====================================================================
st.subheader("绿电套餐")

g1, g2 = st.columns([1, 1])
with g1:
    green_ratio = st.slider("绿电比例", 0.0, 1.0, 0.20, 0.05)
with g2:
    green_fix_price = st.number_input(
        "绿电价 (元/kWh)",
        0.0000, 2.0000, 0.0200,
        step=0.0001, format="%.4f"
    )

if p_green_avg > 0:
    ratio_to_avg = green_fix_price / p_green_avg * 100
    st.caption(f"当前绿电价为批发绿电均值的 {ratio_to_avg:.2f}%")
    if ratio_to_avg > 110:
        st.warning(
            f"⚠️ 当前绿电价 {green_fix_price:.4f} 元/kWh 高于批发均值 "
            f"{p_green_avg:.4f} 的 110%，需市场风险提示。"
        )

# =====================================================================
# 执行收益计算
# =====================================================================
params_profit = {
    "k1_ratio": k1_ratio,
    "k2_ratio": k2_ratio,
    "k3_ratio": k3_ratio,
    "k2_float_percent": k2_float_percent,
    "market_avg_price": market_avg_price,
    "k3_share_ratio": k3_share_ratio,
}
params_profit, warn_list = enforce_rules_profit(params_profit)

results_profit = calc_profit_detailed(
    total_power_mwh, wholesale_price,
    k1_curve, params_profit["k1_ratio"],
    params_profit["market_avg_price"], params_profit["k2_float_percent"], params_profit["k2_ratio"],
    k3_input_curve, k3_is_factor, params_profit["k3_share_ratio"], params_profit["k3_ratio"],
    green_ratio, green_fix_price,
)

st.subheader("收益结果")
for w in warn_list:
    st.warning(w)
st.json(results_profit)

# =====================================================================
# 市场均价敏感性分析
# =====================================================================
st.subheader("敏感性分析：市场均价 ±30%")

changes = np.arange(-0.3, 0.31, 0.05)
sens_rows = []
for c in changes:
    r = calc_profit_detailed(
        total_power_mwh, wholesale_price,
        k1_curve, params_profit["k1_ratio"],
        params_profit["market_avg_price"] * (1 + c), params_profit["k2_float_percent"], params_profit["k2_ratio"],
        k3_input_curve, k3_is_factor, params_profit["k3_share_ratio"], params_profit["k3_ratio"],
        green_ratio, green_fix_price,
    )
    sens_rows.append([c * 100, r["单位收益(元/kWh)"]])

df_sens = pd.DataFrame(sens_rows, columns=["市场均价变动(%)", "单位收益(元/kWh)"]).set_index("市场均价变动(%)")
st.line_chart(df_sens)

