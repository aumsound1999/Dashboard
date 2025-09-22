# app.py
# -*- coding: utf-8 -*-
import ast
import re
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Data loading (เชื่อมกับของเดิม)
# =========================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    โหลดตารางสแนปช็อต per channel 'ชั่วโมงล่าสุด'
    ต้องคืน DataFrame ที่อย่างน้อยมีคอลัมน์:
        - 'channel' หรือ 'C_name' (ชื่อช่อง)
        - 1 คอลัมน์ฝั่ง 'ข้อมูลแอดรายแคมป์' (มีสตริงที่มี 'gmv:' / 'auto:' และลิสต์แคมเปญ)
        - 1 คอลัมน์ฝั่ง 'สแนปช็อตรวม' (สตริงตัวเลขคั่น , สิ้นสุดด้วยค่าเครดิต)
    หมายเหตุ: ด้านล่างมี auto-detect ชื่อคอลัมน์ให้ ถ้าชื่อไม่ตรงก็ยังเดินต่อได้
    """
    # TODO: เชื่อมกับแหล่งข้อมูลจริงของคุณ (Google Sheet/DB/CSV)
    # ที่นี่ใส่ dummy โครงสร้างไว้ให้เห็นหน้าตาเท่านั้น
    data = [
        {
            "channel": "aaa",
            "ads_blob": "('gmv:u0s2.0', 'auto:u0s1.0', [['ro_30', 1250, 34, 4, 95, 720, 20.89], ['ro_99', 900, 0, 0, 0, 0, 0.0]])",
            "snapshot": "2025,12,34,776,22,51,400"
        },
        {
            "channel": "bbb",
            "ads_blob": "('gmv:u0s2.0', 'auto:u0s2.0', [['camp_a', 800, 0, 0, 0, 0, 0.0], ['camp_b', 700, 0, 0, 0, 0, 0.0]])",
            "snapshot": "2025,11,22,500,10,25,455"
        },
        {
            "channel": "ccc",
            "ads_blob": "('gmv:u0s2.0', 'auto:u0s1.0', [['c1', 1000, 25, 2, 33, 195, 25.34]])",
            "snapshot": "2025,10,22,333,11,22,444"
        },
        {
            "channel": "ddd",
            "ads_blob": "('gmv:u0s2.0', 'auto:u0s1.0', [['c1', 1000, 0, 0, 0, 0, 0.0]])",
            "snapshot": "2025,10,22,333,11,22,444"
        },
    ]
    return pd.DataFrame(data)


# =========================
# Helpers: auto-detect columns
# =========================
def find_channel_col(df: pd.DataFrame) -> str:
    cand = [c for c in df.columns if c.lower() in ("channel", "c_name", "cname")]
    if cand:
        return cand[0]
    # ถ้าไม่มี ให้เดาว่าคอลัมน์แรกคือชื่อช่อง
    return df.columns[0]


def find_ads_blob_col(df: pd.DataFrame) -> Optional[str]:
    """
    หา column ที่เก็บสตริงเหมือน ('gmv:u0s2.0', 'auto:u0s1.0', [[...]])
    """
    for col in df.columns:
        if df[col].dtype == object:
            s = str(df[col].dropna().astype(str).head(3).tolist())
            if "gmv:" in s and "auto:" in s and "[[" in s:
                return col
    return None


def find_snapshot_col(df: pd.DataFrame) -> Optional[str]:
    """
    หา column snapshot ที่เป็นตัวเลขคั่นด้วย , และเราอยากใช้ตัวท้ายสุดเป็น credit
    """
    pattern = re.compile(r"^\s*\d+(,\s*\d+){3,}\s*$")
    for col in df.columns:
        if df[col].dtype == object:
            vals = df[col].dropna().astype(str).head(5)
            ok_count = sum(bool(pattern.match(v)) for v in vals)
            if ok_count >= max(1, len(vals) // 2):
                return col
    return None


# =========================
# Parse functions
# =========================
def parse_campaign_blob(v: str) -> List[List[float]]:
    """
    พาร์สก้อนแคมเปญฝั่งแอด
    Expected literal_eval -> (gmv_flag, auto_flag, [[name, budget, spend, orders, views, gmv, roas], ...])
    คืนลิสต์ของแคมเปญ (list)
    """
    try:
        tup = ast.literal_eval(v)
        if isinstance(tup, tuple) and len(tup) >= 3 and isinstance(tup[2], list):
            return tup[2]
        # บางเคสข้อมูลเป็น list เดี่ยว
        if isinstance(tup, list) and len(tup) and isinstance(tup[0], list):
            return tup
    except Exception:
        pass
    return []


def count_active_campaigns(v: str) -> int:
    """
    นับแคมเปญที่ 'เปิด' แบบ conservative = spend > 0
    """
    camps = parse_campaign_blob(v)
    active = 0
    for item in camps:
        # รูปแบบคาดหวัง: [name, budget, spend, orders, views, gmv, roas]
        if len(item) >= 3:
            try:
                spend = float(item[2])
                if spend > 0:
                    active += 1
            except Exception:
                continue
    return active


def extract_credit_from_snapshot(s: str) -> Optional[float]:
    """
    ดึงเครดิตคงเหลือจาก snapshot (เอาตัวท้ายสุด)
    ตัวอย่าง: '2025,12,34,776,22,51,1036' -> 1036
    """
    if not isinstance(s, str):
        return None
    try:
        parts = [x.strip() for x in s.split(",")]
        if not parts:
            return None
        return float(parts[-1])
    except Exception:
        return None


# =========================
# UI Blocks
# =========================
def render_low_credit_panel(df_latest: pd.DataFrame):
    """
    วาดบล็อก Advertising credits are low
    - คัดช่องที่เครดิต < 500 และมีแคมเปญเปิดอยู่ > 0
    """
    channel_col = find_channel_col(df_latest)
    ads_col = find_ads_blob_col(df_latest)
    snap_col = find_snapshot_col(df_latest)

    if ads_col is None or snap_col is None:
        st.info("ไม่พบคอลัมน์ข้อมูลฝั่งแอดหรือสแนปช็อตรวม จึงไม่สามารถคัดช่องเครดิตต่ำได้")
        return

    out_rows: List[Tuple[str, float, int]] = []
    for _, row in df_latest.iterrows():
        ch = str(row[channel_col])

        credit = extract_credit_from_snapshot(row.get(snap_col, ""))
        if credit is None:
            continue

        n_active = count_active_campaigns(str(row.get(ads_col, "")))

        if (credit < 500) and (n_active > 0):
            out_rows.append((ch, credit, n_active))

    if not out_rows:
        st.success("ทุกช่องมีเครดิตเพียงพอ หรือยังไม่มีแคมเปญที่เปิดอยู่ 😊")
        return

    show = pd.DataFrame(out_rows, columns=["ช่อง", "เครดิตคงเหลือ", "แคมเปญที่เปิด"])
    show = show.sort_values(["เครดิตคงเหลือ", "แคมเปญที่เปิด"], ascending=[True, False]).reset_index(drop=True)

    st.markdown("### Advertising credits are low")
    st.caption("แสดงเฉพาะช่องที่เครดิตคงเหลือน้อยกว่า 500 และมีแคมเปญที่เปิดใช้งานอยู่ (spend > 0)")
    st.dataframe(
        show.style.format({"เครดิตคงเหลือ": lambda x: f"{int(x):,}", "แคมเปญที่เปิด": "{:d}"}),
        use_container_width=True,
        hide_index=True,
    )


# =========================
# App
# =========================
st.set_page_config(page_title="Shopee ROAS Dashboard", layout="wide")

# Filters (เหมือนของเดิม – คุณเชื่อมจริงตามโปรเจกต์)
with st.sidebar:
    st.header("Filters")
    st.button("Reload")

# โหลดข้อมูล
df_latest = load_data()

# KPI ส่วนบน (ของเดิมคุณมีแล้ว—คงโครงสร้างไว้/ดึงจาก df_latest ตามจริง)
kpi_cols = st.columns(5)
with kpi_cols[0]:
    st.metric("Sales", f"{int(np.random.randint(150000, 250000)):,}", "+34.0%")
with kpi_cols[1]:
    st.metric("Orders", f"{int(np.random.randint(800, 1300)):,}", "+28.0%")
with kpi_cols[2]:
    st.metric("Budget (Ads)", f"{int(np.random.randint(6000, 9000)):,}", "+9.0%")
with kpi_cols[3]:
    st.metric("sale_ro (Sales/Ads)", f"{np.random.uniform(20, 35):.3f}", "+21.0%")
with kpi_cols[4]:
    st.metric("ads_ro (avg>0)", f"{np.random.uniform(15, 25):.2f}", "-12.0%")

st.markdown("## Trend overlay by day")
st.caption("*ส่วนกราฟเดิมของคุณคงไว้ (ไม่เปลี่ยน)*")

# -------------------------
# แทนที่ “Data (hourly latest snapshot per channel)”
# -------------------------
st.markdown("---")
render_low_credit_panel(df_latest)
st.markdown("---")

st.markdown("## Prime hours heatmap")
st.caption("*ส่วน Heatmap เดิมของคุณคงไว้ (ไม่เปลี่ยน)*")

# หมายเหตุ:
# 1) หน้า Channel – Time series table (hourly) ไม่ถูกแตะ (คุณใช้โค้ดเดิมต่อไป)
# 2) ถ้าในโปรเจกต์จริง ชื่อคอลัมน์ไม่ตรง โค้ดนี้ auto-detect ให้แล้ว
# 3) ถ้าต้องใช้นิยาม 'แคมเปญเปิด' ที่ละเอียดขึ้น เปลี่ยน logic ในฟังก์ชัน count_active_campaigns() ได้เลย
