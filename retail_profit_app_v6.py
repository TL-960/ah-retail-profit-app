import streamlit as st
import numpy as np
import pandas as pd

# =====================================================================
# 一、各省默认现货价格曲线（单位：元/MWh）
# =====================================================================

# 安徽默认现货（24 点）
DEFAULT_SPOT_AH_24 = np.array([
    370.10, 363.18, 360.90, 353.45, 351.18, 360.30,
    363.12, 335.49, 259.44, 215.67, 190.45, 168.96,
    155.90, 171.09, 202.53, 259.66, 334.17, 370.44,
    376.52, 374.20, 372.50, 370.44, 358.88, 364.50
], dtype=float)

# 福建默认现货（24 点）
DEFAULT_SPOT_FJ_24 = np.array([
    348.5, 290.1, 286.9, 250.4, 232.2, 262.7,
    273.8, 223.8, 222.9, 221.8, 227.2, 198.9,
    189.3, 227.2, 234.6, 217.7, 282.1, 277.3,
    305.8, 289.7, 308.4, 293.2, 313.4, 269.7
], dtype=float)

# 山东默认现货（24 点）
DEFAULT_SPOT_SD_24 = np.array([
    372.0818179, 359.8755787, 346.7290448, 334.8613549,
    334.2911806, 346.5936451, 330.3188873, 269.2929738,
    173.6974105, 109.1121883, 101.6097778, 97.14061574,
    82.14963426, 97.63985957, 145.7927222, 222.9569938,
    353.8797176, 444.7974090, 462.9419846, 454.2064398,
    447.0688040, 437.3273086, 411.8231775, 397.8648179
], dtype=float)

# 浙江默认现货（48 点半小时）
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
# 二、安徽配置（24 点 TOU + 用户曲线分配中长期）
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
    return user_curve * ratio

# =====================================================================
# 三、福建配置（24 点 TOU + 第一曲线 + 谷电比 + 分摊价）
# =====================================================================

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

FIRST_CURVE_FJ_RAW = np.array([
    4.28, 4.24, 4.17, 4.12, 4.08, 4.05,
    4.06, 4.13, 4.30, 4.36, 4.25, 4.04,
    4.00, 4.22, 4.30, 4.28, 4.32, 4.18,
    4.13, 4.10, 4.13, 4.05, 4.11, 4.11
], dtype=float)
FIRST_CURVE_FJ = FIRST_CURVE_FJ_RAW / FIRST_CURVE_FJ_RAW.sum()

def make_contract_curve_fj(total_user_mwh, ratio):
    return total_user_mwh * ratio * FIRST_CURVE_FJ

def calc_valley_ratio_fj(user_curve, month):
    tags = TOU_TABLE_FJ[month]
    valley_hours = [i for i, t in enumerate(tags) if t == "谷"]
    valley_energy = user_curve[valley_hours].sum()
    total_energy = user_curve.sum()
    if total_energy <= 0:
        return 0.0
    return valley_energy / total_energy

# =====================================================================
# 四、浙江配置（典型负荷曲线 48 点 + 48 点 TOU）
# =====================================================================

# 典型负荷：1–11 月（48 点，占比）
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

# 典型负荷：12 月（48 点，占比）
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

# 春秋季 TOU（2–6 月、9–11 月，48 点）
TOU_ZJ_SPRING_AUTUMN_48 = [
    # 00:00–07:00 → 谷（14 半小时：0-13）
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

# 夏冬季 TOU（1、7、8、12 月，48 点）
TOU_ZJ_SUMMER_WINTER_48 = [
    # 00:00–07:00 → 谷
    "谷","谷","谷","谷","谷","谷","谷",
    "谷","谷","谷","谷","谷","谷","谷",
    # 07:00–11:00 → 平
    "平","平","平","平","平","平","平","平",
    # 11:00–14:00 → 谷
    "谷","谷","谷","谷","谷","谷",
    # 14:00–16:00 → 平
    "平","平","平","平",
    # 16:00–18:00 → 峰
    "峰","峰","峰","峰",
    # 18:00–22:00 → 尖
    "尖","尖","尖","尖","尖","尖","尖","尖",
    # 22:00–23:00 → 峰
    "峰","峰",
    # 23:00–24:00 → 平
    "平","平"
]

def get_tou_zj_by_month(month: int):
    if month in [2, 3, 4, 5, 6, 9, 10, 11]:
        return TOU_ZJ_SPRING_AUTUMN_48
    else:  # 1, 7, 8, 12
        return TOU_ZJ_SUMMER_WINTER_48

def split_load_by_tou_zj(month, 尖, 峰, 平, 谷):
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
    shape = ZJ_LOAD_DEC_48 if month == 12 else ZJ_LOAD_JAN_NOV_48
    return total_user_mwh * ratio * shape

# =====================================================================
# 五、山东配置（24 点 TOU + 深谷 + 直线典型曲线）
# =====================================================================

# 直线典型曲线：24 个点均为 1/24
SD_TYPICAL_24 = np.full(24, 1.0 / 24.0, dtype=float)

SD_TOU_COL_A = [  # 1-2, 12 月
    "平","平","谷","谷","谷","谷","平",
    "峰","峰","平","谷","深谷","深谷","深谷",
    "谷","平",
    "尖峰","尖峰","尖峰",
    "峰","峰",
    "平","平","平"
]

SD_TOU_COL_B = [  # 3-5 月
    "平","平","平","平","平","平","平",
    "平","平","平","谷","深谷","深谷","深谷",
    "谷","平","平",
    "尖峰","尖峰","尖峰",
    "峰","峰",
    "平","平"
]

SD_TOU_COL_C = [  # 6 月
    "平","平","平","平","平","平","平",
    "谷","谷","谷","谷","谷","平","平",
    "平","平","峰",
    "尖峰","尖峰","尖峰",
    "尖峰","尖峰",
    "峰","平"
]

SD_TOU_COL_D = [  # 7-8 月
    "平","谷","谷","谷","谷","谷",
    "平","平","平","平","平","平","平",
    "平","平","平",
    "峰","尖峰","尖峰",
    "尖峰","尖峰",
    "尖峰","峰","平"
]

SD_TOU_COL_E = [  # 9-11 月
    "平","平","平","平","平","平","平",
    "平","平","平","谷","深谷","深谷","深谷",
    "谷","平",
    "峰","尖峰","尖峰",
    "峰","峰",
    "平","平","平"
]

def get_tou_sd_by_month(month: int):
    """山东：返回对应月份的 24 点 TOU（含：尖峰、峰、平、谷、深谷）"""
    if month in [1, 2, 12]:
        return SD_TOU_COL_A
    elif month in [3, 4, 5]:
        return SD_TOU_COL_B
    elif month == 6:
        return SD_TOU_COL_C
    elif month in [7, 8]:
        return SD_TOU_COL_D
    else:  # 9-11
        return SD_TOU_COL_E

def split_load_by_tou_sd(month, 尖峰电量, 峰电量, 平电量, 谷电量, 深谷电量):
    """
    山东：根据 24 点 TOU 把尖峰/峰/平/谷/深谷月电量拆成 24 点曲线（MWh）
    """
    tags = get_tou_sd_by_month(month)
    cnt_尖峰 = tags.count("尖峰")
    cnt_峰 = tags.count("峰")
    cnt_平 = tags.count("平")
    cnt_谷 = tags.count("谷")
    cnt_深谷 = tags.count("深谷")

    curve = []
    for tag in tags:
        if tag == "尖峰":
            curve.append(尖峰电量 / cnt_尖峰 if cnt_尖峰 > 0 else 0.0)
        elif tag == "峰":
            curve.append(峰电量 / cnt_峰 if cnt_峰 > 0 else 0.0)
        elif tag == "平":
            curve.append(平电量 / cnt_平 if cnt_平 > 0 else 0.0)
        elif tag == "谷":
            curve.append(谷电量 / cnt_谷 if cnt_谷 > 0 else 0.0)
        elif tag == "深谷":
            curve.append(深谷电量 / cnt_深谷 if cnt_深谷 > 0 else 0.0)
        else:
            curve.append(0.0)
    return np.array(curve, dtype=float)

def make_contract_curve_sd(total_user_mwh, ratio):
    """山东：中长期曲线 = 总电量 × 比例 × 直线典型曲线"""
    return total_user_mwh * ratio * SD_TYPICAL_24

# =====================================================================
# 六、通用白天用电比例计算（9:00–15:00）
# =====================================================================

def calc_daytime_ratio(user_curve):
    """
    白天用电比例（9:00–15:00）
    24 点：取 index 9~14 共 6 点
    48 点：取 index 18~29 共 12 半小时点
    """
    n = len(user_curve)
    total = float(np.sum(user_curve))
    if total <= 0:
        return 0.0

    if n == 24:
        day_energy = float(np.sum(user_curve[9:15]))
    elif n == 48:
        day_energy = float(np.sum(user_curve[18:30]))
    else:
        raise ValueError(f"不支持的曲线点数：{n}")

    return day_energy / total


# =====================================================================
# 七、通用成本计算函数（支持 24 或 48 点，单位：MWh & 元/MWh）
# =====================================================================

def calc_cost(user_curve,
              contract_curve,
              long_price_curve,
              spot_price_curve,
              allocation_price=None):
    """
    user_curve, contract_curve: MWh 曲线
    long_price_curve, spot_price_curve: 元/MWh 曲线
    allocation_price: 分摊价（元/MWh），如为 None 则不计分摊
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
# 八、Streamlit UI
# =====================================================================

st.set_page_config(page_title="多省电力零售成本测算（皖/闽/浙/鲁）", layout="wide")
st.title("⚡ 多省电力零售成本测算（安徽 / 福建 / 浙江 / 山东，单位：元/MWh）")

province = st.selectbox("选择省份", ["安徽", "福建", "浙江", "山东"], index=0)

# ---------- 现货价格输入（按省份分支） ----------
if province in ["安徽", "福建", "山东"]:
    st.subheader("一、日前现货价格曲线（24 点整点，单位：元/MWh）")

    if province == "安徽":
        base_spot = DEFAULT_SPOT_AH_24
    elif province == "福建":
        base_spot = DEFAULT_SPOT_FJ_24
    else:  # 山东
        base_spot = DEFAULT_SPOT_SD_24

    use_custom_spot_24 = st.checkbox(
        "手动输入 24 点现货价格（元/MWh）", value=False, key=f"spot24_ck_{province}"
    )
    if use_custom_spot_24:
        vals = base_spot.copy().tolist()
        for r in range(4):
            cols = st.columns(6)
            for c in range(6):
                i = r * 6 + c
                vals[i] = cols[c].number_input(
                    f"{i}:00",
                    0.0, 10000.0, float(vals[i]),
                    step=0.0001,
                    format="%.4f",
                    key=f"spot24_{province}_{i}"
                )
        spot_curve_24 = np.array(vals, dtype=float)
    else:
        spot_curve_24 = base_spot.copy()

else:
    st.subheader("一、日前现货价格曲线（48 点半小时，单位：元/MWh）")
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
# 安徽模块
# =====================================================================
if province == "安徽":
    st.header("二、安徽成本精算模块（24 点）")

    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("月份（用于 TOU 拆分）", list(range(1, 13)), index=5)
    P_long = c2.number_input(
        "中长期合同电价（元/MWh）",
        0.0, 5000.0, 360.0000,
        step=0.0001, format="%.4f"
    )
    ratio = c3.number_input(
        "中长期比例（0~1）",
        0.0, 1.0, 0.8800,
        step=0.0001, format="%.4f"
    )

    e1, e2, e3, e4 = st.columns(4)
    尖 = e1.number_input("尖电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    峰 = e2.number_input("峰电量 (MWh)", 0.0, 1e9, 3475.1200, step=0.0001, format="%.4f")
    平 = e3.number_input("平电量 (MWh)", 0.0, 1e9, 1559.3600, step=0.0001, format="%.4f")
    谷 = e4.number_input("谷电量 (MWh)", 0.0, 1e9, 14809.5200, step=0.0001, format="%.4f")

    user_curve = split_load_by_tou_ah(month, 尖, 峰, 平, 谷)
    contract_curve = make_contract_curve_ah(user_curve, ratio)
    long_price_curve = np.full(24, P_long, dtype=float)
    spot_curve = spot_curve_24

    res = calc_cost(user_curve, contract_curve, long_price_curve, spot_curve)

    day_ratio = calc_daytime_ratio(user_curve)
    st.write(f"🌞 白天用电占比（9–15 点）：**{day_ratio * 100:.4f}%**")

    st.success(f"📌 安徽：平均购电成本 = **{res['final_avg']:.4f} 元/MWh**")

    with st.expander("展开查看成本明细（安徽）"):
        ratio_real = contract_curve.sum() / max(user_curve.sum(), 1e-9)
        st.write(f"- 总用电量：{res['total_mwh']:.4f} MWh")
        st.write(f"- 实际中长期电量占比：{ratio_real*100:.4f}%")
        st.write(f"- 中长期成本：{res['long_cost']:,.4f} 元")
        st.write(f"- 现货偏差成本：{res['spot_cost']:,.4f} 元")
        st.write(f"- 总购电成本：{res['final_total_cost']:,.4f} 元")

    df = pd.DataFrame({
        "用户用电(MWh)": user_curve,
        "中长期(MWh)": contract_curve,
        "偏差电量(MWh)": res["dev_curve"],
    }, index=[f"{i}:00" for i in range(24)])
    st.line_chart(df)

# =====================================================================
# 福建模块
# =====================================================================
elif province == "福建":
    st.header("二、福建成本精算模块（第一曲线 + 谷电比 + 分摊价，24 点）")

    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("月份（用于 TOU 拆分）", list(range(1, 13)), index=5)
    P_long = c2.number_input(
        "中长期合同电价（元/MWh）",
        0.0, 5000.0, 360.0000,
        step=0.0001, format="%.4f"
    )
    ratio = c3.number_input(
        "中长期比例（0~1，例如 0.80）",
        0.0, 1.0, 0.8000,
        step=0.0001, format="%.4f"
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
            step=0.0001, format="%.4f"
        )

    res = calc_cost(user_curve, contract_curve, long_price_curve,
                    spot_curve, allocation_price)

    day_ratio = calc_daytime_ratio(user_curve)
    st.write(f"🌞 白天用电占比（9–15 点）：**{day_ratio * 100:.4f}%**")

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
# 浙江模块
# =====================================================================
elif province == "浙江":
    st.header("二、浙江成本精算模块（48 点，典型负荷 + 分摊价）")

    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("月份（用于 TOU 拆分 & 典型曲线选择）", list(range(1, 13)), index=11)
    P_long = c2.number_input(
        "中长期合同电价（元/MWh）",
        0.0, 5000.0, 360.0000,
        step=0.0001, format="%.4f"
    )
    ratio = c3.number_input(
        "中长期比例（0~1，例如 0.90）",
        0.0, 1.0, 0.9000,
        step=0.0001, format="%.4f"
    )

    e1, e2, e3, e4 = st.columns(4)
    尖 = e1.number_input("尖电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    峰 = e2.number_input("峰电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    平 = e3.number_input("平电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    谷 = e4.number_input("谷电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")

    user_curve = split_load_by_tou_zj(month, 尖, 峰, 平, 谷)
    total_user_mwh = float(user_curve.sum())

    contract_curve = make_contract_curve_zj(total_user_mwh, ratio, month)
    long_price_curve = np.full(48, P_long, dtype=float)
    spot_curve = spot_curve_48

    use_allocation = st.checkbox("启用分摊价格（元/MWh）", value=False)
    allocation_price = None
    if use_allocation:
        allocation_price = st.number_input(
            "分摊单价（元/MWh）",
            0.0, 1000.0, 10.0000,
            step=0.0001, format="%.4f"
        )

    res = calc_cost(user_curve, contract_curve, long_price_curve,
                    spot_curve, allocation_price)

    day_ratio = calc_daytime_ratio(user_curve)
    st.write(f"🌞 白天用电占比（9–15 点）：**{day_ratio * 100:.4f}%**")

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

    time_index_48 = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]
    df = pd.DataFrame({
        "用户用电(MWh)": user_curve,
        "中长期(MWh)": contract_curve,
        "偏差电量(MWh)": res["dev_curve"],
    }, index=time_index_48)
    st.line_chart(df)

# =====================================================================
# 山东模块
# =====================================================================
else:  # 山东
    st.header("二、山东成本精算模块（尖峰/峰/平/谷/深谷 + 直线典型曲线，24 点）")

    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("月份（用于 TOU 拆分）", list(range(1, 13)), index=5)
    P_long = c2.number_input(
        "中长期合同电价（元/MWh）",
        0.0, 5000.0, 360.0000,
        step=0.0001, format="%.4f"
    )
    ratio = c3.number_input(
        "中长期比例（0~1，例如 0.85）",
        0.0, 1.0, 0.8500,
        step=0.0001, format="%.4f"
    )

    e1, e2, e3, e4, e5 = st.columns(5)
    尖峰 = e1.number_input("尖峰电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    峰 = e2.number_input("峰电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    平 = e3.number_input("平电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    谷 = e4.number_input("谷电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")
    深谷 = e5.number_input("深谷电量 (MWh)", 0.0, 1e9, 0.0, step=0.0001, format="%.4f")

    user_curve = split_load_by_tou_sd(month, 尖峰, 峰, 平, 谷, 深谷)
    total_user_mwh = float(user_curve.sum())

    contract_curve = make_contract_curve_sd(total_user_mwh, ratio)
    long_price_curve = np.full(24, P_long, dtype=float)
    spot_curve = spot_curve_24

    use_allocation = st.checkbox("启用分摊价格（元/MWh）", value=False)
    allocation_price = None
    if use_allocation:
        allocation_price = st.number_input(
            "分摊单价（元/MWh）",
            0.0, 1000.0, 10.0000,
            step=0.0001, format="%.4f"
        )

    res = calc_cost(user_curve, contract_curve, long_price_curve,
                    spot_curve, allocation_price)

    day_ratio = calc_daytime_ratio(user_curve)
    st.write(f"🌞 白天用电占比（9–15 点）：**{day_ratio * 100:.4f}%**")

    if allocation_price is None:
        st.success(
            f"📌 山东：基础平均购电成本（不含分摊） = **{res['base_avg']:.4f} 元/MWh**"
        )
    else:
        st.success(
            f"📌 山东：基础平均购电成本 = **{res['base_avg']:.4f} 元/MWh**；"
            f"分摊价 = **{allocation_price:.4f} 元/MWh**；"
            f"最终平均购电成本 = **{res['final_avg']:.4f} 元/MWh**"
        )

    with st.expander("展开查看成本明细（山东）"):
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
