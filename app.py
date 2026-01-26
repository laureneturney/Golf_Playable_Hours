import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
import io
import sys

# --- CONFIGURATION & SETTINGS ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = Path.cwd()

CITY_TO_STATION_ID = {
    "Chicago": "USW00094846",
    "Cincinnati": "USW00093814",
    "Kansas City": "USW00003947",
    "Minneapolis": "USW00014922",
    "Boston": "USW00014739",
    "DC": "USW00093738",
    "Denver": "USW00093067",
    "Seattle": "USW00024234",
    "San Francisco": "USW00023234",
    "Salt Lake City": "USW00024127",
    "Atlanta": "USW00013874",
    "Miami": "USW00012839",
    "Tampa": "USW00012842",
    "Dallas": "USW00003927",
    "Houston": "USW00012960",
    "Phoenix": "USW00023183",
    "San Diego": "USW00023188",
    "Madison": "USW00014837",
    'New York': "USW00094789"
}

# --- RESTORED HELPER FUNCTIONS FROM OLD VERSION ---

def time_to_float(t):
    if pd.isna(t): return np.nan
    try:
        t = str(t).strip()
        if t == "" or t.lower() == "nan": return np.nan
        if ":" in t:
            parts = t.split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            s = int(parts[2]) if len(parts) > 2 else 0
            return h + (m / 60) + (s / 3600)
        return float(t)
    except Exception: return np.nan

def fetch_noaa_lcd_cloud(station_id: str, year: int) -> pd.DataFrame:
    url = f"https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/{year}/LCD_{station_id}_{year}.csv"
    try:
        df = pd.read_csv(url, low_memory=False, encoding='utf-8')
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df[["DATE", "HourlyDryBulbTemperature", "HourlyWindSpeed", 
                   "HourlyPrecipitation", "HourlyPresentWeatherType", "HourlyVisibility"]]
    except Exception: return None

def load_sunrise_sunset_cloud(station_id: str) -> tuple[pd.DataFrame, dict]:
    csv_path = BASE_DIR / "SUNRISE_SUNSET" / f"{station_id}_SUNRISE_SUNSET.csv"
    if not csv_path.exists(): return None, None
    sun = pd.read_csv(csv_path, encoding='utf-8')
    sun.columns = [c.strip() for c in sun.columns]
    sun["Date"] = pd.to_datetime(sun["Date"], errors="coerce").dt.normalize()
    sun["sr_float"] = sun["SUNRISE_LST"].apply(time_to_float)
    sun["ss_float"] = sun["SUNSET_LST"].apply(time_to_float)
    # Restore legacy 12-hour PM convention
    sun["ss_float"] = sun["ss_float"].where(sun["ss_float"] <= 12, sun["ss_float"] - 12)
    sun_lookup = sun.dropna(subset=["Date", "sr_float", "ss_float"]).set_index("Date")[["sr_float", "ss_float"]].to_dict("index")
    return sun, sun_lookup

def classify_rain(code: str, precip: float) -> float:
    code = (code or "").upper()
    p = float(precip) if pd.notna(precip) else np.nan
    if pd.isna(p) and code == "": return np.nan
    if ("+RA" in code) or ("+TS" in code) or (pd.notna(p) and p >= 0.25): return 2.0
    if (("RA" in code) and ("-RA" not in code)) or (pd.notna(p) and p >= 0.10): return 1.0
    if ("-RA" in code) or ("DZ" in code) or (pd.notna(p) and p > 0): return 0.5
    return 0.0

def compute_rain_blocked_cons(severity: pd.Series) -> pd.Series:
    blocked, heavy_streak, post_block_remaining = [], 0, 0
    for sev in severity.values:
        if pd.isna(sev):
            blocked.append(False); heavy_streak = 0; post_block_remaining = 0
            continue
        if sev == 2.0:
            heavy_streak += 1; post_block_remaining = 0; blocked.append(True)
        else:
            if heavy_streak >= 2 and post_block_remaining == 0: post_block_remaining = 3
            heavy_streak = 0
            if post_block_remaining > 0:
                blocked.append(True); post_block_remaining -= 1
            else: blocked.append(False)
    return pd.Series(blocked, index=severity.index)

# --- CORE GPH ALGORITHM (RESTORED FROM ORIGINAL) ---

def run_city_gph(city_caps: str, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_key = city_caps.title() if city_caps != "DC" else "DC"
    station_id = CITY_TO_STATION_ID[city_key]
    
    all_weather_dfs = [fetch_noaa_lcd_cloud(station_id, yr) for yr in years]
    weather_df = pd.concat([df for df in all_weather_dfs if df is not None], ignore_index=True)
    _, sun_lookup = load_sunrise_sunset_cloud(station_id)

    # 1. Reindexing & DATA_OK
    weather_df["Hour_Rounded"] = weather_df["DATE"].dt.floor("H")
    hourly = weather_df.drop_duplicates("Hour_Rounded", keep="last").set_index("Hour_Rounded")
    hourly["DATA_OK"] = 1
    full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="H")
    hourly = hourly.reindex(full_idx)
    hourly["DATA_OK"] = hourly["DATA_OK"].fillna(0).astype(int)

    # 2. Cleaning & Interpolation
    def clean_to_float(s): return pd.to_numeric(s.astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce")
    def interp(s): return s.interpolate(method="time", limit_direction="both").ffill().bfill()

    hourly["Temp_F"] = interp(clean_to_float(hourly["HourlyDryBulbTemperature"]) * 9/5 + 32)
    hourly["Wind_Speed"] = interp(clean_to_float(hourly["HourlyWindSpeed"]))
    hourly["Visibility_Mi"] = interp(clean_to_float(hourly["HourlyVisibility"]))
    
    hourly["Precip"] = clean_to_float(hourly["HourlyPrecipitation"].astype(str).str.strip().replace("T", "0.001"))
    hourly.loc[hourly["DATA_OK"] == 1, "Precip"] = hourly.loc[hourly["DATA_OK"] == 1, "Precip"].fillna(0.0)

    # 3. Weather Flags
    wc = hourly["HourlyPresentWeatherType"].fillna("").astype(str).str.upper()
    hourly["Has_Thunder"] = wc.str.contains(r"\bTS\b|\bTSTM\b|\bTSRA\b|LTG", regex=True)
    hourly["Has_Wet_Code"] = wc.str.contains(r"\bRA\b|\bDZ\b|\bBR\b|\bFG\b", regex=True)
    hourly["Wet_LowVis"] = (hourly["Has_Wet_Code"]) & (hourly["Visibility_Mi"] < 6)

    # 4. Daylight (Corrected sr-0.5 / ss+0.5)
    def get_daylight_flag(ts):
        dt = ts.normalize()
        if dt not in sun_lookup: return 0
        hr = ts.hour + ts.minute / 60.0
        sr, ss = sun_lookup[dt]["sr_float"] - 0.5, sun_lookup[dt]["ss_float"] + 0.5
        return 1 if (sr <= hr <= ss) else 0
    hourly["Daylight_Flag"] = [get_daylight_flag(ts) for ts in hourly.index]

    # 5. Rain Severity & Blocked logic
    hourly["Rain_Severity"] = [classify_rain(c, p) for c, p in zip(wc, hourly["Precip"])]
    hourly["Rain_Blocked_CONS"] = compute_rain_blocked_cons(hourly["Rain_Severity"])

    # 6. GPH Formulas (RESTORED EXACTLY)
    # All flags REQUIRE (DATA_OK == 1)
    hourly["GPH_OPT"] = ((hourly["DATA_OK"] == 1) & (hourly["Daylight_Flag"] == 1) & (hourly["Temp_F"] >= 32) & (hourly["Wind_Speed"] <= 30) & (~hourly["Has_Thunder"]) & (hourly["Rain_Severity"] <= 1.0)).astype(int)
    
    hourly["GPH_BASE"] = ((hourly["DATA_OK"] == 1) & (hourly["Daylight_Flag"] == 1) & (hourly["Temp_F"] >= 38) & (hourly["Wind_Speed"] <= 25) & (~hourly["Has_Thunder"]) & (~hourly["Wet_LowVis"]) & (hourly["Rain_Severity"] <= 1.0)).astype(int)
    
    hourly["GPH_CONS"] = ((hourly["DATA_OK"] == 1) & (hourly["Daylight_Flag"] == 1) & (hourly["Temp_F"] >= 40) & (hourly["Wind_Speed"] <= 22) & (~hourly["Has_Thunder"]) & (~hourly["Wet_LowVis"]) & (hourly["Rain_Severity"] <= 0.5) & (~hourly["Rain_Blocked_CONS"])).astype(int)

    summary = hourly.groupby(hourly.index.year)[["GPH_OPT", "GPH_BASE", "GPH_CONS"]].sum()
    return hourly, summary

# --- STREAMLIT UI ---

def main():
    st.set_page_config(page_title="Cloud GPH Calculator", layout="wide")
    st.title("🏙️ National GPH Calculation Tool (Cloud Edition)")
    st.info("Algorithms Restored to Original Version. Fetching Live NOAA Data.")

    col1, col2 = st.columns(2)
    with col1:
        available_cities = sorted([c.upper() for c in CITY_TO_STATION_ID.keys()])
        select_all = st.checkbox("Select All Cities")
        selected_cities = st.multiselect("Select Cities", available_cities, default=available_cities if select_all else [])
        year_input = st.text_input("Enter Years (comma separated)", value="2024")

    if st.button("Calculate GPH", use_container_width=True):
        years = [int(y.strip()) for y in year_input.split(",") if y.strip()]
        rows = []
        progress = st.progress(0)
        for i, city in enumerate(selected_cities):
            with st.spinner(f"Processing {city}..."):
                try:
                    _, annual = run_city_gph(city, years)
                    for yr, row in annual.iterrows():
                        rows.append({"CITY": city, "YEAR": yr, "GPH_OPT": int(row["GPH_OPT"]), "GPH_BASE": int(row["GPH_BASE"]), "GPH_CONS": int(row["GPH_CONS"])})
                except Exception as e: st.error(f"Error {city}: {e}")
            progress.progress((i + 1) / len(selected_cities))

        if rows:
            final_df = pd.DataFrame(rows).sort_values(["CITY", "YEAR"])
            st.success("Analysis Complete!")
            st.dataframe(final_df, use_container_width=True)
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results CSV", data=csv, file_name="GPH_Analysis.csv", mime="text/csv")

if __name__ == "__main__":
    main()
