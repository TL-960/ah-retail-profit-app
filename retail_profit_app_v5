import streamlit as st
import numpy as np
import pandas as pd

# =====================================================================
# 通用：安徽/福建默认现货曲线（单位：元/MWh，24 点）
# =====================================================================
DEFAULT_SPOT_MWH_24 = np.array([
    370.10, 363.18, 360.90, 353.45, 351.18, 360.30,
    363.12, 335.49, 259.44, 215.67, 190.45, 168.96,
    155.90, 171.09, 202.53, 259.66, 334.17, 370.44,
    376.52, 374.20, 372.50, 370.44, 358.88, 364.50
], dtype=float)

# =====================================================================
# 浙江默认现货曲线（单位：元/MWh，48 点，半小时）
# =====================================================================
DEFAULT_SPOT_ZJ_48 = np.array([
    343.8865895, 343.1267009, 341.6607349, 339.9209611, 342.4790429,
    332.0358448, 325.7966506, 318.5811972, 313.5679028, 314.1470074,
    322.1215062, 329.2280105, 333.2762108, 322.9830182, 324.00665,
    315.3226985, 289.0344769, 280.8211361, 278.5530503, 273.5722367,
    278.5313244, 261.2193796, 297.6305716, 262.4512549, 289.5421617,
    292.8177108, 250.8009593, 267.6961873, 281.3856707, 301.184104,
    304.5998099, 312.1248272, 331.5506664, 330.8062244, 338.4700025,
    341.9855059, 344.5440772, 341.1427522, 341.730216, 339.1469247,
    343.3799648, 348.7415256, 342.207491, 331.0865154, 337.1445605,
    350.0255731, 348.4160281, 333.8894145
], dtype=float)

# =====================================================================
# 一、安徽配置（官方 TOU + 按用户曲线分配中长期）——24 点
# =====================================================================

# 冬季（1 月、12 月）
WINTER_TOU_AH = [
    "谷","谷","谷","谷","谷","谷",   # 0-5
    "平","平","平","平","平","平",   # 6-11
    "谷","谷",                       # 12-13
    "平",                            # 14
    "峰","峰","峰","峰","峰","峰","峰","峰",  # 15-22
    "谷"                             # 23
]

# 春秋季（2-6 月、10-11 月）
SPRING_AUTUMN_TOU_AH = [
    "谷","谷","谷","谷","谷","谷",       # 0-5
    "峰","峰",                           # 6-7
    "平","平","平",                      # 8-10
    "谷","谷","谷",                      # 11-13
    "平","平",                           # 14-15
    "峰","峰","峰","峰","峰","峰",       # 16-21
    "平",                                # 22
    "谷"                                 # 23
]

# 夏季（7-9 月）
SUMMER_TOU_AH = [
    "平","平",                            # 0-1
    "谷","谷","谷","谷","谷","谷","谷",    # 2-8
    "平","平",                            # 9-10
    "谷","谷",                            # 11-12
    "平","平","平",                       # 13-15
    "峰","峰","峰","峰","峰","峰","峰","峰"  # 16-23
]

TOU_TABLE_AH = {
    1: WINTER_TOU_AH,
    2: SPRING_AUTUMN_TOU_AH,
    3: SPRING_AUTUMN_TOU_AH,
    4: SPRING_AUTUMN_TOU_AH,
    5: SPRING_AUTUMN_TOU_AH,
    6: SPRING_AUTUMN_TOU_AH,
    7: SUMMER_TOU_AH,
    8: SUMMER_TOU_AH,
    9: SUMMER_TOU_AH,
    10: SPRING_AUTUMN_TOU_AH,
    11: SPRING_AUTUMN_TOU_AH,
    12: WINTER_TOU_AH,
}

def split_load_by_tou_ah(month, 尖, 峰, 平, 谷):
    """安徽：根据 TOU_TABLE_AH 把尖峰平谷月电量拆成 24 点曲线（MWh）"""
    tags = TOU_TABLE_AH[month]
    cnt_尖 = tags.count("尖")
    cnt_峰 = tags.count("峰")
    cnt_平 = tags.count("平")
    cnt_谷 = tags.count("谷")

    curve = []
    for tag in tags:
        if tag == "尖":
            curve.append(尖 / cnt_尖 if cnt_尖 > 0 else 0.0)
        elif tag == "峰":
            curve.append(峰 / cnt_峰 if cnt_峰 > 0 else 0.0)
        elif tag == "平":
            curve.append(平 / cnt_平 if cnt_平 > 0 else 0.0)
        elif tag == "谷":
            curve.append(谷 / cnt_谷 if cnt_谷 > 0 else 0.0)
        else:
            curve.append(0.0)
    return np.array(curve, dtype=float)

def make_contract_curve_ah(user_curve, ratio):
    """安徽：中长期曲线 = 用户曲线 × 比例"""
    return user_curve * ratio

# =====================================================================
# 二、福建配置（TOU + 第一曲线 + 谷电比 + 分摊价）——24 点
# =====================================================================

# 1-6 月 & 10-12 月 TOU（非夏季）
TOU_FJ_NON_SUMMER = [
    "谷","谷","谷","谷","谷","谷","谷","谷",  # 0-7
    "平","平",                                # 8-9
    "峰","峰",                                # 10-11
    "平","平","平",                           # 12-14
    "峰","峰","峰","峰","峰",                  # 15-19
    "平",                                     # 20
    "峰",                                     # 21
    "平","平"                                 # 22-23
]

# 7-9 月 TOU（夏季，含尖峰）
TOU_FJ_SUMMER = [
    "谷","谷","谷","谷","谷","谷","谷","谷",   # 0-7
    "平","平",                                 # 8-9
    "峰","尖",                                 # 10-11
    "平","平","平",                            # 12-14
    "峰","峰","尖","峰","峰",                  # 15-19
    "平",                                      # 20
    "峰",                                      # 21
    "平","平"                                  # 22-23
]

TOU_TABLE_FJ = {
    m: (TOU_FJ_SUMMER if m in [7, 8, 9] else TOU_FJ_NON_SUMMER)
    for m in range(1, 13)
}

def split_load_by_tou_fj(month, 尖, 峰, 平, 谷):
    """福建：根据 TOU_TABLE_FJ 把尖峰平谷月电量拆成 24 点用户曲线（MWh）"""
    tags = TOU_TABLE_FJ[month]
    cnt_尖 = tags.count("尖")
    cnt_峰 = tags.count("峰")
    cnt_平 = tags.count("平")
    cnt_谷 = tags.count("谷")

    curve = []
    for tag in tags:
        if tag == "尖":
            curve.append(尖 / cnt_尖 if cnt_尖 > 0 else 0.0)
        elif tag == "峰":
            curve.append(峰 / cnt_峰 if cnt_峰 > 0 else 0.0)
        elif tag == "平":
            curve.append(平 / cnt_平 if cnt_平 > 0 else 0.0)
        elif tag == "谷":
            curve.append(谷 / cnt_谷 if cnt_谷 > 0 else 0.0)
        else:
            curve.append(0.0)
    return np.array(curve, dtype=float)

# 福建第一曲线（24 点百分比 → 占比）
FIRST_CURVE_FJ_RAW = np.array([
    4.28, 4.24, 4.17, 4.12, 4.08, 4.05,
    4.06, 4.13, 4.30, 4.36, 4.25, 4.04,
    4.00, 4.22, 4.30, 4.28, 4.32, 4.18,
    4.13, 4.10, 4.13, 4.05, 4.11, 4.11
], dtype=float)
FIRST_CURVE_FJ = FIRST_CURVE_FJ_RAW / FIRST_CURVE_FJ_RAW.sum()  # 和=1

def make_contract_curve_fj(total_user_mwh, ratio):
    """福建：中长期曲线 = 总电量 × 比例 × 第一曲线占比（MWh）"""
    return total_user_mwh * ratio * FIRST_CURVE_FJ

def calc_valley_ratio_fj(user_curve, month):
    """福建谷电比 = 谷段电量 / 总电量（按福建 TOU 的“谷”小时求和）"""
    tags = TOU_TABLE_FJ[month]
    valley_hours = [i for i, t in enumerate(tags) if t == "谷"]
    valley_energy = user_curve[valley_hours].sum()
    total_energy = user_curve.sum()
    if total_energy <= 0:
        return 0.0
    return valley_energy / total_energy

# =====================================================================
# 三、浙江配置（48 点 TOU + 典型负荷曲线）
# =====================================================================

# --- 典型负荷曲线：1–11 月（48 点，占比） ---
ZJ_LOAD_JAN_NOV_48_RAW = np.array([
    0.019318, 0.018629, 0.018042, 0.017565,
    0.017173, 0.017173, 0.016639, 0.016467,
    0.016377, 0.016375, 0.016475, 0.016867,
    0.017441, 0.018507, 0.019913, 0.021629,
    0.022693, 0.023260, 0.023275, 0.023038,
    0.022811, 0.022578, 0.021742, 0.020980,
    0.020556, 0.021017, 0.021042, 0.020744,
    0.020779, 0.020978, 0.021324, 0.021909,
    0.022624, 0.023240, 0.023058, 0.023476,
    0.023777, 0.023864, 0.023828, 0.023771,
    0.023562, 0.023486, 0.023155, 0.022705,
    0.022250, 0.022199, 0.021385, 0.020304
], dtype=float)
ZJ_LOAD_JAN_NOV_48 = ZJ_LOAD_JAN_NOV_48_RAW / ZJ_LOAD_JAN_NOV_48_RAW.sum()

# --- 典型负荷曲线：12 月（48 点，占比） ---
ZJ_LOAD_DEC_48_RAW = np.array([
    0.018877, 0.018642, 0.018193, 0.017830,
    0.017526, 0.017526, 0.016997, 0.016788,
    0.016662, 0.016574, 0.016659, 0.017049,
    0.017632, 0.018791, 0.020198, 0.021801,
    0.022728, 0.023048, 0.023030, 0.022745,
    0.022535, 0.022309, 0.021476, 0.021252,
    0.020929, 0.021680, 0.021775, 0.021509,
    0.021629, 0.021888, 0.022222, 0.022588,
    0.023137, 0.023523, 0.023145, 0.023381,
    0.023409, 0.023412, 0.023228, 0.023034,
    0.022726, 0.022589, 0.022240, 0.021806,
    0.021346, 0.021367, 0.020747, 0.019937
], dtype=float)
ZJ_LOAD_DEC_48 = ZJ_LOAD_DEC_48_RAW / ZJ_LOAD_DEC_48_RAW.sum()

# --- 浙江 TOU：春秋季（2–6 月、9–11 月）48 点 ---
# 高峰：16:00–23:00
# 平段：7:00–11:00、14:00–16:00、23:00–24:00
# 低谷：0:00–7:00、11:00–14:00
TOU_ZJ_SPRING_AUTUMN_48 = [
    # 00:00–07:00 → 低谷（14 半小时：0-13）
    "谷","谷","谷","谷","谷","谷","谷",
    "谷","谷","谷","谷","谷","谷","谷",
    # 07:00–11:00 → 平（14-21）
    "平","平","平","平","平","平","平","平",
    # 11:00–14:00 → 谷（22-27）
    "谷","谷","谷","谷","谷","谷",
    # 14:00–16:00 → 平（28-31）
    "平","平","平","平",
    # 16:00–23:00 → 峰（32-45）
    "峰","峰","峰","峰","峰","峰","峰",
    "峰","峰","峰","峰","峰","峰","峰",
    # 23:00–24:00 → 平（46-47）
    "平","平"
]

# --- 浙江 TOU：夏冬季（1、7、8、12 月）48 点 ---
# 高峰：16:00–18:00、22:00–23:00
# 平段：7:00–11:00、14:00–16:00、23:00–24:00
# 低谷：0:00–7:00、11:00–14:00
# 尖峰：18:00–22:00
TOU_ZJ_SUMMER_WINTER_48 = [
    # 00:00–07:00 → 谷（0-13）
    "谷","谷","谷","谷","谷","谷","谷",
    "谷","谷","谷","谷","谷","谷","谷",
    # 07:00–11:00 → 平（14-21）
    "平","平","平","平","平","平","平","平",
    # 11:00–14:00 → 谷（22-27）
    "谷","谷","谷","谷","谷","谷",
    # 14:00–16:00 → 平（28-31）
    "平","平","平","平",
    # 16:00–18:00 → 峰（32-35）
    "峰","峰","峰","峰",
    # 18:00–22:00 → 尖（36-43）
    "尖","尖","尖","尖","尖","尖","尖","尖",
    # 22:00–23:00 → 峰（44-45）
    "峰","峰",
    # 23:00–24:00 → 平（46-47）
    "平","平"
]

def get_tou_zj_by_month(month: int):
    """浙江：根据月份返回对应 48 点 TOU"""
    if month in [2, 3, 4, 5, 6, 9, 10, 11]:
        return TOU_ZJ_SPRING_AUTUMN_48
    else:  # 1, 7, 8, 12
        return TOU_ZJ_SUMMER_WINTER_48

def split_load_by_tou_zj(month, 尖, 峰, 平, 谷):
    """浙江：根据 48 点 TOU 把尖峰平谷月电量拆成 48 点曲线（MWh）"""
    tags = get_tou_zj_by_month(month)
    cnt_尖 = tags.count("尖")
    cnt_峰 = tags.count("峰")
    cnt_平 = tags.count("平")
    cnt_谷 = tags.count("谷")

    curve = []
    for tag in tags:
        if tag == "尖":
            curve.append(尖 / cnt_尖 if cnt_尖 > 0 else 0.0)
        elif tag == "峰":
            curve.append(峰 / cnt_峰 if cnt_峰 > 0 else 0.0)
        elif tag == "平":
            curve.append(平 / cnt_平 if cnt_平 > 0 else 0.0)
        elif tag == "谷":
            curve.append(谷 / cnt_谷 if cnt_谷 > 0 else 0.0)
        else:
            curve.append(0.0)
    return np.array(curve, dtype=float)

def make_contract_curve_zj(total_user_mwh, ratio, month):
    """浙江：中长期曲线 = 总电量 × 比例 × 对应典型负荷曲线占比（MWh，48 点）"""
    shape = ZJ_LOAD_DEC_48 if month == 12 else ZJ_LOAD_JAN_NOV_48
    return total_user_mwh * ratio * shape

# =====================================================================
# 四、通用成本计算函数（长度可为 24 或 48，全部使用 MWh & 元/MWh）
# =====================================================================

def calc_cost(user_curve,
              contract_curve,
              long_price_curve,
              spot_price_curve,
              allocation_price=None):
    """
    user_curve, contract_curve: MWh 曲线（长度 n）
    long_price_curve, spot_price_curve: 元/MWh 曲线（长度 n）
    allocation_price: 分摊价（元/MWh），如为 None 则不计分摊

    返回：
    - 所有成本单位：元
    - 平均成本单位：元/MWh
    """
    n = len(user_curve)
    if not (len(contract_curve) == len(long_price_curve) == len(spot_price_curve) == n):
        raise ValueError(
            f"曲线长度不一致：user={len(user_curve)}, contract={len(contract_curve)}, "
            f"long_price={len(long_price_curve)}, spot_price={len(spot_price_curve)}，应全部相等。"
        )

    dev = user_curve - contract_curve  # MWh

    long_cost = float(np.sum(contract_curve * long_price_curve))
    spot_cost = float(np.sum(dev * spot_price_curve))
    base_total_cost = long_cost + spot_cost

    total_mwh = float(user_curve.sum())
    base_avg = base_total_cost / max(total_mwh, 1e-9)  # 元/MWh

    allocation_cost = 0.0
    if allocation_price is not None:
        allocation_cost = total_mwh * allocation_price
        final_total_cost = base_total_cost + allocation_cost
        final_avg = base_avg + allocation_price
    else:
        final_total_cost = base_total_cost
        final_avg = base_avg

    return {
        "dev_curve": dev,
        "long_cost": long_cost,
        "spot_cost": spot_cost,
        "base_total_cost": base_total_cost,
        "base_avg": base_avg,
        "allocation_cost": allocation_cost,
        "final_total_cost": final_total_cost,
        "final_avg": final_avg,
        "total_mwh": total_mwh,
    }

# =====================================================================
# 五、Streamlit UI（安徽 + 福建 + 浙江，全部元/MWh）
# =====================================================================

st.set_page_config(page_title="多省电力零售成本测算（皖/闽/浙）", layout="wide")
st.title("⚡ 多省电力零售成本测算（安徽 / 福建 / 浙江）")

province = st.selectbox("选择省份", ["安徽", "福建", "浙江"], index=0)

# ---------- 现货价格输入（按省份分支） ----------
if province in ["安徽", "福建"]:
    st.subheader("一、日前现货价格曲线（单位：元/MWh，24 点整点）")
    use_custom_spot_24 = st.checkbox(
        "手动输入 24 点现货价格（元/MWh）", value=False, key="spot24_ck"
    )
    if use_custom_spot_24:
        vals = DEFAULT_SPOT_MWH_24.copy().tolist()
        for r in range(4):
            cols = st.columns(6)
            for c in range(6):
                i = r * 6 + c
                vals[i] = cols[c].number_input(
                    f"{i}:00",
                    0.0, 10000.0, float(vals[i]),
                    step=0.0001,
                    format="%.4f",
                    key=f"spot24_{i}"
                )
        spot_curve_24 = np.array(vals, dtype=float)
    else:
        spot_curve_24 = DEFAULT_SPOT_MWH_24.copy()

else:
    st.subheader("一、日前现货价格曲线（单位：元/MWh，48 点半小时）")
    use_custom_spot_48 = st.checkbox(
        "手动输入 48 点现货价格（元/MWh）", value=False, key="spot48_ck"
    )
    if use_custom_spot_48:
        vals = DEFAULT_SPOT_ZJ_48.copy().tolist()
        for r in range(8):
            cols = st.columns(6)
            for c in range(6):
                i = r * 6 + c
                hh = i // 2
                mm = (i % 2) * 30
                vals[i] = cols[c].number_input(
                    f"{hh:02d}:{mm:02d}",
                    0.0, 10000.0, float(vals[i]),
                    step=0.0001,
                    format="%.4f",
                    key=f"spot48_{i}"
                )
        spot_curve_48 = np.array(vals, dtype=float)
    else:
        spot_curve_48 = DEFAULT_SPOT_ZJ_48.copy()

st.divider()

# =====================================================================
# 安徽模块（24 点）
# =====================================================================
if province == "安徽":
    st.header("二、安徽成本精算模块（24 点，单位：元/MWh）")

    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("月份（用于 TOU 拆分）", list(range(1, 13)), index=5)
    P_long = c2.number_input(
        "中长期合同电价（元/MWh）",
        0.0, 5000.0, 360.0000,
        step=0.0001,
        format="%.4f"
    )
    ratio = c3.number_input(
        "中长期比例（0~1）",
        0.0, 1.0, 0.8800,
        step=0.0001,
        format="%.4f"
    )

    e1, e2, e3, e4 = st.columns(4)
    尖 = e1.number_input("尖电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    峰 = e2.number_input("峰电量 (MWh)", 0.0, 1e9, 3475.1200, step=0.0001, format="%.4f")
    平 = e3.number_input("平电量 (MWh)", 0.0, 1e9, 1559.3600, step=0.0001, format="%.4f")
    谷 = e4.number_input("谷电量 (MWh)", 0.0, 1e9, 14809.5200, step=0.0001, format="%.4f")

    user_curve = split_load_by_tou_ah(month, 尖, 峰, 平, 谷)
    contract_curve = make_contract_curve_ah(user_curve, ratio)
    long_price_curve = np.full(24, P_long, dtype=float)
    spot_curve = spot_curve_24  # 24 点

    res = calc_cost(user_curve, contract_curve, long_price_curve, spot_curve)

    st.success(f"📌 安徽：平均购电成本 = **{res['final_avg']:.4f} 元/MWh**")

    with st.expander("展开查看成本明细（安徽）"):
        ratio_real = contract_curve.sum() / max(user_curve.sum(), 1e-9)
        st.write(f"- 总用电量：{res['total_mwh']:.4f} MWh")
        st.write(f"- 实际中长期电量占比：{ratio_real*100:.4f}%")
        st.write(f"- 中长期成本：{res['long_cost']:,.4f} 元")
        st.write(f"- 现货偏差成本：{res['spot_cost']:,.4f} 元")
        st.write(f"- 基础总购电成本（不含分摊）：{res['base_total_cost']:,.4f} 元")
        if res["allocation_cost"] > 0:
            st.write(f"- 分摊成本：{res['allocation_cost']:,.4f} 元")
        st.write(f"- 最终总购电成本：{res['final_total_cost']:,.4f} 元")

    df = pd.DataFrame({
        "用户用电(MWh)": user_curve,
        "中长期(MWh)": contract_curve,
        "偏差电量(MWh)": res["dev_curve"],
    }, index=[f"{i}:00" for i in range(24)])
    st.line_chart(df)

# =====================================================================
# 福建模块（24 点）
# =====================================================================
elif province == "福建":
    st.header("二、福建成本精算模块（第一曲线 + 谷电比 + 分摊价，24 点）")

    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("月份（用于 TOU 拆分）", list(range(1, 13)), index=5)
    P_long = c2.number_input(
        "中长期合同电价（元/MWh）",
        0.0, 5000.0, 360.0000,
        step=0.0001,
        format="%.4f"
    )
    ratio = c3.number_input(
        "中长期比例（0~1，例如 0.80）",
        0.0, 1.0, 0.8000,
        step=0.0001,
        format="%.4f"
    )

    e1, e2, e3, e4 = st.columns(4)
    尖 = e1.number_input("尖电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    峰 = e2.number_input("峰电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    平 = e3.number_input("平电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    谷 = e4.number_input("谷电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")

    user_curve = split_load_by_tou_fj(month, 尖, 峰, 平, 谷)
    total_user_mwh = float(user_curve.sum())

    contract_curve = make_contract_curve_fj(total_user_mwh, ratio)
    long_price_curve = np.full(24, P_long, dtype=float)
    spot_curve = spot_curve_24

    use_allocation = st.checkbox("启用分摊价格（元/MWh）", value=False)
    allocation_price = None
    if use_allocation:
        allocation_price = st.number_input(
            "分摊单价（元/MWh）",
            0.0, 1000.0, 10.0000,
            step=0.0001,
            format="%.4f"
        )

    res = calc_cost(user_curve, contract_curve, long_price_curve,
                    spot_curve, allocation_price)

    valley_ratio = calc_valley_ratio_fj(user_curve, month)

    if allocation_price is None:
        st.success(
            f"📌 福建：基础平均购电成本（不含分摊） = **{res['base_avg']:.4f} 元/MWh**；"
            f"🌙 谷电比 = **{valley_ratio*100:.4f}%**"
        )
    else:
        st.success(
            f"📌 福建：基础平均购电成本 = **{res['base_avg']:.4f} 元/MWh**；"
            f"分摊价 = **{allocation_price:.4f} 元/MWh**；"
            f"最终平均购电成本 = **{res['final_avg']:.4f} 元/MWh**；"
            f"🌙 谷电比 = **{valley_ratio*100:.4f}%**"
        )

    with st.expander("展开查看成本明细（福建）"):
        ratio_real = contract_curve.sum() / max(user_curve.sum(), 1e-9)
        st.write(f"- 总用电量：{res['total_mwh']:.4f} MWh")
        st.write(f"- 实际中长期电量占比：{ratio_real*100:.4f}%")
        st.write(f"- 中长期成本：{res['long_cost']:,.4f} 元")
        st.write(f"- 现货偏差成本：{res['spot_cost']:,.4f} 元")
        st.write(f"- 基础总购电成本（不含分摊）：{res['base_total_cost']:,.4f} 元")
        if allocation_price is not None:
            st.write(f"- 分摊成本：{res['allocation_cost']:,.4f} 元")
        st.write(f"- 最终总购电成本：{res['final_total_cost']:,.4f} 元")

    df = pd.DataFrame({
        "用户用电(MWh)": user_curve,
        "中长期(MWh)": contract_curve,
        "偏差电量(MWh)": res["dev_curve"],
    }, index=[f"{i}:00" for i in range(24)])
    st.line_chart(df)

# =====================================================================
# 浙江模块（48 点）
# =====================================================================
else:
    st.header("二、浙江成本精算模块（48 点，典型负荷 + 分摊价）")

    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("月份（用于 TOU 拆分 & 典型曲线选择）", list(range(1, 13)), index=11)
    P_long = c2.number_input(
        "中长期合同电价（元/MWh）",
        0.0, 5000.0, 360.0000,
        step=0.0001,
        format="%.4f"
    )
    ratio = c3.number_input(
        "中长期比例（0~1，例如 0.90）",
        0.0, 1.0, 0.9000,
        step=0.0001,
        format="%.4f"
    )

    e1, e2, e3, e4 = st.columns(4)
    尖 = e1.number_input("尖电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    峰 = e2.number_input("峰电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    平 = e3.number_input("平电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    谷 = e4.number_input("谷电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")

    # 用户曲线（按浙江 48 点 TOU 拆分）
    user_curve = split_load_by_tou_zj(month, 尖, 峰, 平, 谷)
    total_user_mwh = float(user_curve.sum())

    # 中长期曲线（按典型负荷曲线 + 比例）
    contract_curve = make_contract_curve_zj(total_user_mwh, ratio, month)
    long_price_curve = np.full(48, P_long, dtype=float)

    # 现货价格：采用浙江默认 48 点或用户自定义
    spot_curve = spot_curve_48

    # 分摊价
    use_allocation = st.checkbox("启用分摊价格（元/MWh）", value=False)
    allocation_price = None
    if use_allocation:
        allocation_price = st.number_input(
            "分摊单价（元/MWh）",
            0.0, 1000.0, 10.0000,
            step=0.0001,
            format="%.4f"
        )

    res = calc_cost(user_curve, contract_curve, long_price_curve,
                    spot_curve, allocation_price)

    if allocation_price is None:
        st.success(
            f"📌 浙江：基础平均购电成本（不含分摊） = **{res['base_avg']:.4f} 元/MWh**"
        )
    else:
        st.success(
            f"📌 浙江：基础平均购电成本 = **{res['base_avg']:.4f} 元/MWh**；"
            f"分摊价 = **{allocation_price:.4f} 元/MWh**；"
            f"最终平均购电成本 = **{res['final_avg']:.4f} 元/MWh**"
        )

    with st.expander("展开查看成本明细（浙江）"):
        ratio_real = contract_curve.sum() / max(user_curve.sum(), 1e-9)
        st.write(f"- 总用电量：{res['total_mwh']:.4f} MWh")
        st.write(f"- 实际中长期电量占比：{ratio_real*100:.4f}%")
        st.write(f"- 中长期成本：{res['long_cost']:,.4f} 元")
        st.write(f"- 现货偏差成本：{res['spot_cost']:,.4f} 元")
        st.write(f"- 基础总购电成本（不含分摊）：{res['base_total_cost']:,.4f} 元")
        if allocation_price is not None:
            st.write(f"- 分摊成本：{res['allocation_cost']:,.4f} 元")
        st.write(f"- 最终总购电成本：{res['final_total_cost']:,.4f} 元")

    # 48 点时间索引（0:00, 0:30, ..., 23:30）
    time_index_48 = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

    df = pd.DataFrame({
        "用户用电(MWh)": user_curve,
        "中长期(MWh)": contract_curve,
        "偏差电量(MWh)": res["dev_curve"],
    }, index=time_index_48)
    st.line_chart(df)
