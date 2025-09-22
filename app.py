# ================== LOW-CREDIT PANEL (Drop-in, Single-file) ==================
# วางโค้ดบล็อกนี้ไว้ด้านบนไฟล์ app.py / หน้า Overview เดิมได้เลย

from __future__ import annotations
import re
from typing import Optional, Tuple, List

import pandas as pd
import streamlit as st


# -------------------- CONFIG --------------------
LOW_CREDIT_THRESHOLD = 500  # เครดิตคงเหลือต่ำกว่าเท่านี้ ให้แจ้งเตือน
SECTION_TITLE = "Advertising credits are low"
LEFT_LABEL  = "ช่อง"
MID1_LABEL  = "เครดิตเหลือ"
MID2_LABEL  = "แคมป์ที่เปิด"
RIGHT_LABEL = "ช่อง"
# ------------------------------------------------


def _find_channel_col(df: pd.DataFrame) -> str:
    """พยายามหา column ชื่อช่อง: 'channel' หรือ 'c_name' หรือคอลัมน์แรก"""
    candidates = [c for c in df.columns if str(c).strip().lower() in ("channel", "c_name")]
    if candidates:
        return candidates[0]
    return df.columns[0]


def _looks_like_ads_blob(series: pd.Series) -> bool:
    """เดาว่าคอลัมน์นี้เป็นก้อนข้อมูลแอด (มี 'gmv:' และ 'auto:')"""
    cnt = 0
    for v in series.astype(str).head(50):
        s = v.replace(" ", "")
        if "gmv:" in s and "auto:" in s:
            cnt += 1
    return cnt >= max(1, len(series) // 50)  # มีบ้างถือว่าใช่


def _find_ads_blob_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        s = df[c]
        if s.dtype == "object" and _looks_like_ads_blob(s):
            return c
    return None


def _looks_like_snapshot(series: pd.Series) -> bool:
    """
    เดาว่าเป็นคอลัมน์ snapshot:
    - เป็นสตริงของตัวเลขคั่นด้วย comma เช่น '2025,12,34,776,22,51,1036'
    - ค่าสุดท้ายคือเครดิตคงเหลือ (ตัวเลข)
    """
    ok, tried = 0, 0
    for v in series.astype(str).head(50):
        tried += 1
        parts = [p for p in v.split(",") if p.strip() != ""]
        if len(parts) >= 2 and re.fullmatch(r"[0-9]+", parts[-1]) is not None:
            ok += 1
    return ok >= max(1, tried // 3)


def _find_snapshot_col(df: pd.DataFrame) -> Optional[str]:
    candidates = []
    for c in df.columns:
        if df[c].dtype == "object" and _looks_like_snapshot(df[c]):
            candidates.append(c)
    if not candidates:
        return None
    # ถ้ามีหลายคอลัมน์ เลือกคอลัมน์ที่ *ขวาสุด* (ล่าสุด)
    rightmost = None
    right_idx = -1
    for c in candidates:
        idx = list(df.columns).index(c)
        if idx > right_idx:
            right_idx, rightmost = idx, c
    return rightmost


def _safe_int(x) -> Optional[int]:
    """แปลงเป็น int อย่างปลอดภัย"""
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _extract_credit_from_snapshot(val: str) -> Optional[int]:
    """
    value snapshot -> เครดิตคงเหลือ (ตัวสุดท้าย)
    '2025,12,34,776,22,51,1036' -> 1036
    """
    if pd.isna(val):
        return None
    parts = [p for p in str(val).split(",") if p.strip() != ""]
    if not parts:
        return None
    return _safe_int(parts[-1])


def _extract_active_campaigns_from_ads_blob(val: str) -> int:
    """
    ก้อนข้อมูลแอด -> นับจำนวนแคมป์ที่ 'เปิดใช้งาน'
    กติกา: ในส่วนรายการแคมป์ (ลิสต์ด้านท้าย) เราถือว่า 'เปิด' เมื่อยอดใช้จ่าย (field ที่ 2) > 0
    ตัวอย่าง: ['ro _30', 1250, 34, 4, 95, 720, 20.89] -> 34 คือ 'กินเงิน' > 0 ถือว่าเปิด
    """
    if pd.isna(val):
        return 0
    s = str(val)

    # ตัดส่วนสรุปสิทธิ 'gmv:...' 'auto:...' ออก (ถ้ามี)
    parts = s.split("]]")
    tail = s
    if len(parts) >= 2:
        tail = parts[-1]

    # ดึงกลุ่มที่อยู่ในวงเล็บเหลี่ยม เช่น [ 'ro', 1250, 34, ... ]
    bracket_groups = re.findall(r"\[(.*?)\]", tail)
    cnt = 0
    for g in bracket_groups:
        # split comma แต่ระวังเครื่องหมาย quote
        toks = [t.strip(" '\"") for t in g.split(",")]
        # ต้องมีอย่างน้อย 3 ช่อง: name, budget, spend
        if len(toks) >= 3:
            spend = _safe_int(toks[2])
            if spend is not None and spend > 0:
                cnt += 1
    return cnt


def _build_low_credit_table(df_latest: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    ย่อย df_latest ให้เหลือเฉพาะช่องที่:
    - เครดิตคงเหลือ < LOW_CREDIT_THRESHOLD
    - มีแคมป์ที่เปิดใช้งาน > 0
    ผลลัพธ์: DataFrame 4 คอลัมน์ (ซ้าย-กลาง-ขวา) สำหรับใช้โชว์ 2 คอลัมน์/แถว
    """
    if df_latest is None or df_latest.empty:
        return None

    chan_col = _find_channel_col(df_latest)
    ads_col  = _find_ads_blob_col(df_latest)
    snap_col = _find_snapshot_col(df_latest)

    if ads_col is None or snap_col is None:
        return None

    work = df_latest[[chan_col, ads_col, snap_col]].copy()
    work.rename(columns={chan_col: "channel", ads_col: "ads_blob", snap_col: "snapshot"}, inplace=True)

    work["credit_left"] = work["snapshot"].apply(_extract_credit_from_snapshot)
    work["active_camps"] = work["ads_blob"].apply(_extract_active_campaigns_from_ads_blob)

    filt = (work["credit_left"].notna()) & (work["credit_left"] < LOW_CREDIT_THRESHOLD) & (work["active_camps"] > 0)
    work = work.loc[filt, ["channel", "credit_left", "active_camps"]].copy()

    if work.empty:
        return None

    work.sort_values(["credit_left", "channel"], inplace=True, ignore_index=True)

    # แปลงเป็นรูป 2 คอลัมน์ (ซ้าย/ขวา) เพื่อโชว์สวย ๆ
    rows = []
    vals = work.to_dict("records")
    for i in range(0, len(vals), 2):
        left  = vals[i]
        right = vals[i+1] if i+1 < len(vals) else None
        rows.append({
            LEFT_LABEL:  left["channel"],
            MID1_LABEL:  left["credit_left"],
            MID2_LABEL:  f'{left["active_camps"]} แคม',
            RIGHT_LABEL: right["channel"] if right else ""
        })
    return pd.DataFrame(rows)


def render_low_credit_panel(df_latest: pd.DataFrame) -> None:
    """
    วาง call ฟังก์ชันนี้ ณ ตำแหน่งที่เดิมคุณแสดง Data (hourly latest snapshot per channel)
    """
    st.markdown(f"### {SECTION_TITLE}")

    tbl = _build_low_credit_table(df_latest)
    if tbl is None or tbl.empty:
        st.info("🙌 ไม่มีช่องที่เครดิตต่ำกว่ากำหนด หรือไม่พบคอลัมน์ข้อมูลที่ต้องใช้")
        return

    # แสดง 3 คอลัมน์ (ชื่อช่อง-เครดิต-แคมป์) และคอลัมน์ขวาเป็นชื่อช่อง ถัดไป (เหมือนตัวอย่างที่คุณส่งมา)
    # เลือกแสดงแบบ Markdown เพื่อควบคุมฟอนต์/ระยะห่างให้อ่านง่าย
    # (ถ้าอยากใช้ st.dataframe ก็เปลี่ยนได้)
    cols = st.columns(4)
    cols[0].markdown(f"**{LEFT_LABEL}**")
    cols[1].markdown(f"**{MID1_LABEL}**")
    cols[2].markdown(f"**{MID2_LABEL}**")
    cols[3].markdown(f"**{RIGHT_LABEL}**")

    for _, r in tbl.iterrows():
        cols = st.columns(4)
        cols[0].write(str(r[LEFT_LABEL]))
        cols[1].write(int(r[MID1_LABEL]))
        cols[2].write(str(r[MID2_LABEL]))
        cols[3].write(str(r[RIGHT_LABEL]))

    st.caption("จะแสดงเฉพาะช่องที่เครดิต < %d และมีแคมป์ที่เปิดใช้งานอย่างน้อย 1 แคม" % LOW_CREDIT_THRESHOLD)

# ================== END LOW-CREDIT PANEL ==================
