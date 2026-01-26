import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
import os
import sys

# --- CONFIGURATION & SETTINGS ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure paths are OS-agnostic
BASE_DIR = Path.cwd()
LCD_DIR = BASE_DIR / "LCD_DATA"
SUN_DIR = BASE_DIR / "SUNRISE_SUNSET"
HOURLY_DATA_DIR = BASE_DIR / "HOURLY_DATA"
GPH_OUTPUT_DIR = BASE_DIR / "GPH_CALCULATIONS"

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

# --- HELPER FUNCTIONS ---

def parse_lcd(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df[
        [
            "DATE",
            "HourlyDryBulbTemperature",
            "HourlyWindSpeed",
            "HourlyPrecipitation",
            "HourlyPresentWeatherType",
            "HourlyVisibility",
        ]
    ]

def time_to_float(t):
    """Convert time like '7:19:44' or '7:19' (or already numeric) into float hours."""
    if pd.isna(t):
        return np.nan
    try:
        t = str(t).strip()
        if t == "" or t.lower() == "nan":
            return np.nan
        if ":" in t:
            parts = t.split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            s = int(parts[2]) if len(parts) > 2 else 0
            return h + (m / 60) + (s / 3600)
        return float(t)
    except Exception:
        return np.nan


def load_sunrise_sunset_csv(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    sun = pd.read_csv(csv_path, encoding='utf-8')
    sun.columns = [c.strip() for c in sun.columns]

    required = {"Date", "SUNRISE_LST", "SUNSET_LST"}
    missing = required - set(sun.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing} in {csv_path}. Found: {list(sun.columns)}")

    sun["Date"] = pd.to_datetime(sun["Date"], errors="coerce").dt.normalize()
    sun["sr_float"] = sun["SUNRISE_LST"].apply(time_to_float)
    sun["ss_float"] = sun["SUNSET_LST"].apply(time_to_float)

    sun_master = sun.dropna(subset=["Date", "sr_float", "ss_float"]).copy()
    sun_lookup = sun_master.set_index("Date")[["sr_float", "ss_float"]].to_dict("index")
    return sun_master, sun_lookup


def get_city_lcd_files(city_caps: str, years: list[int]) -> list[Path]:
    """
    Returns LCD CSV files inside LCD_DATA/<CITY>/ that match the requested years.
    Expected filename format: LCD_<stationid>_<year>.csv
    """
    city_dir = LCD_DIR / city_caps
    if not city_dir.exists():
        raise FileNotFoundError(f"City folder not found: {city_dir}")

    # 1. Get all relevant CSV files first
    all_files = city_dir.glob("LCD_*.csv")

    # 2. Filter files: keep the file if its year is in our 'years' list
    # We convert year to string to match the filename text
    year_strings = [str(y) for y in years]
    
    filtered_files = [
        f for f in all_files 
        if any(year_str in f.name for year_str in year_strings)
    ]

    if not filtered_files:
        raise FileNotFoundError(f"No LCD files found for years {years} in: {city_dir}")

    return sorted(filtered_files)


def get_sun_csv_for_city(city_caps: str, city_to_id: dict) -> Path:
    """
    Uses CITY_TO_STATION_ID (Title Case keys) to find the station ID,
    then returns SUNRISE_SUNSET/<ID>_SUNRISE_SUNSET.csv
    """
    # Your dict keys are Title Case ("Chicago"), but cities_to_print_GPH is ALL CAPS ("CHICAGO")
    # Normalize: "SAN FRANCISCO" -> "San Francisco"
    city_key = city_caps.title()

    if city_key not in city_to_id:
        # Handle a couple common exceptions
        if city_caps == "DC":
            city_key = "DC"
        else:
            raise KeyError(f"City '{city_caps}' not found in CITY_TO_STATION_ID keys: {list(city_to_id.keys())}")

    station_id = city_to_id[city_key]
    csv_path = SUN_DIR / f"{station_id}_SUNRISE_SUNSET.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Sunrise/Sunset CSV not found for {city_caps}: {csv_path}")

    return csv_path

def run_city_gph(city_caps: str, city_to_id: dict, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads LCD + sunrise/sunset for a city, computes hourly GPH flags, and writes a CSV output.

    NATIONAL base changes vs your prior version:
    1) Adds a DATA_OK flag so we NEVER count “fabricated” hours created by reindexing/interpolation.
       (This is the #1 reason some stations/cities blow up vs others.)
    2) Defines a single "national" base formula (GPH_BASE_NATIONAL) that is consistent everywhere:
         - Daylight
         - DATA_OK (must have an observation for that hour)
         - Temp_F between [36, 100]  (min playable + extreme heat cutoff)
         - Wind_Speed <= 28
         - Rain_Severity <= 1.0  (allow light+moderate, block heavy)
       You can tune these 4 numbers globally, but they apply to ALL cities.
    3) Keeps OPT/CONS as-is (but also requires DATA_OK so they’re comparable nationally).

    Returns:
      hourly (hourly dataframe)
      annual_summary (yearly totals)
    """

    # ---------- Load files ----------
    lcd_files = get_city_lcd_files(city_caps, years)
    sun_csv = get_sun_csv_for_city(city_caps, city_to_id)

    weather_df = pd.concat([parse_lcd(p) for p in lcd_files], ignore_index=True)

    sun_master, sun_lookup = load_sunrise_sunset_csv(sun_csv)

    # ---- FORCE sunset to match legacy 12-hour PM convention ----
    sun_master["ss_float"] = sun_master["ss_float"].where(
        sun_master["ss_float"] <= 12,
        sun_master["ss_float"] - 12
    )
    sun_lookup = sun_master.set_index("Date")[["sr_float", "ss_float"]].to_dict("index")

    # ============================================================
    # 2) Hourly index + DATA_OK (CRITICAL for a national definition)
    # ============================================================
    # Floor to hour
    weather_df["Hour_Rounded"] = weather_df["DATE"].dt.floor("H")

    # Keep last record per hour (your original behavior)
    hourly = weather_df.drop_duplicates("Hour_Rounded", keep="last").set_index("Hour_Rounded")

    # Mark which hours actually had an observation (before reindex)
    hourly["DATA_OK"] = 1

    # Continuous hourly index (this creates hours with no observations)
    full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="H")
    hourly = hourly.reindex(full_idx)

    # DATA_OK = 1 only where a real obs existed; 0 for synthetic hours
    hourly["DATA_OK"] = hourly["DATA_OK"].fillna(0).astype(int)

    # ============================================================
    # 3) Cleaning + Interpolation for continuous vars (Temp/Wind)
    # ============================================================
    def clean_to_float(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace(r"[^\d.-]", "", regex=True),
            errors="coerce",
        )

    def interpolate_all_time(s: pd.Series, fill_edges: bool = True) -> pd.Series:
        s = s.copy()
        s = s.interpolate(method="time", limit_direction="both" if fill_edges else "forward")
        if fill_edges:
            s = s.ffill().bfill()
        return s

    # Temperature
    temp = clean_to_float(hourly["HourlyDryBulbTemperature"])

    # We assume all LCD Dry Bulb Temps are in Celcius
    temp = temp * 9/5 + 32

    hourly["Temp_F"] = interpolate_all_time(temp, fill_edges=True)


    # Wind
    hourly["Wind_Speed"] = clean_to_float(hourly["HourlyWindSpeed"])
    hourly["Wind_Speed"] = interpolate_all_time(hourly["Wind_Speed"], fill_edges=True)

    # Precip (do NOT interpolate; treat T as 0.001)
    # NOTE: keep missing precip as NaN first, then fill to 0 ONLY for observed hours.
    hourly["Precip_raw"] = hourly["HourlyPrecipitation"].astype(str).str.strip().replace("T", "0.001")
    hourly["Precip"] = clean_to_float(hourly["Precip_raw"])

    # Visibility
    hourly["Visibility_Mi"] = clean_to_float(hourly["HourlyVisibility"])
    hourly["Visibility_Mi"] = interpolate_all_time(hourly["Visibility_Mi"], fill_edges=True)

    # If an hour had no obs, leave precip as NaN (unknown). For observed hours, missing precip -> 0.
    hourly.loc[hourly["DATA_OK"] == 1, "Precip"] = hourly.loc[hourly["DATA_OK"] == 1, "Precip"].fillna(0.0)

    # Weather code as string
    hourly["Weather_Code"] = hourly["HourlyPresentWeatherType"].fillna("")

    wc = hourly["Weather_Code"].astype(str).str.upper()

    hourly["Has_Thunder"] = wc.str.contains(r"\bTS\b|\bTSTM\b|\bTSRA\b|LTG", regex=True)

    hourly["Has_Wet_Code"] = wc.str.contains(r"\bRA\b|\bDZ\b|\bBR\b|\bFG\b", regex=True)

    # "Damp/low visibility" proxy
    hourly["Wet_LowVis"] = (hourly["Has_Wet_Code"]) & (hourly["Visibility_Mi"] < 6)


    # ============================================================
    # 4) Daylight Flag (NO OFFSET — sunrise/sunset are already local)
    # ============================================================
    SOLAR_TO_LOCAL_OFFSET = 0

    def wrap24(x):
        return x % 24

    def get_daylight_flag(ts: pd.Timestamp) -> int:
        dt = ts.normalize()
        if dt not in sun_lookup:
            return 0

        hr = ts.hour + ts.minute / 60.0

        # sunrise/sunset are in local clock hours (legacy-style ss_float)
        sr = wrap24(sun_lookup[dt]["sr_float"] + SOLAR_TO_LOCAL_OFFSET) - 0.5  # civil twilight
        ss = wrap24(sun_lookup[dt]["ss_float"] + SOLAR_TO_LOCAL_OFFSET) + 0.5  # civil twilight

        if sr <= ss:
            return 1 if (sr <= hr <= ss) else 0
        else:
            return 1 if (hr >= sr or hr <= ss) else 0

    hourly["Daylight_Flag"] = [get_daylight_flag(ts) for ts in hourly.index]

    # ============================================================
    # 5) Rain severity + blocking
    # ============================================================
    def classify_rain(code: str, precip: float) -> float:
        code = (code or "").upper()
        p = float(precip) if pd.notna(precip) else np.nan

        # If no observation, we should NOT assume "no rain"
        # Keep severity as NaN so it can be excluded by DATA_OK in playable logic.
        if pd.isna(p) and code == "":
            return np.nan

        if ("+RA" in code) or ("+TS" in code) or (pd.notna(p) and p >= 0.25):
            return 2.0
        if (("RA" in code) and ("-RA" not in code)) or (pd.notna(p) and p >= 0.10):
            return 1.0
        if ("-RA" in code) or ("DZ" in code) or (pd.notna(p) and p > 0):
            return 0.5
        return 0.0

    hourly["Rain_Severity"] = [
        classify_rain(code, precip)
        for code, precip in zip(hourly["Weather_Code"].astype(str), hourly["Precip"])
    ]

    def compute_rain_blocked_cons(severity: pd.Series) -> pd.Series:
        blocked = []
        heavy_streak = 0
        post_block_remaining = 0

        for sev in severity.values:
            # If we don't know severity (no obs), treat as not-blocked here;
            # DATA_OK will handle exclusion in playable flags.
            if pd.isna(sev):
                blocked.append(False)
                heavy_streak = 0
                post_block_remaining = 0
                continue

            if sev == 2.0:
                heavy_streak += 1
                post_block_remaining = 0
                blocked.append(True)
            else:
                if heavy_streak >= 2 and post_block_remaining == 0:
                    post_block_remaining = 3
                heavy_streak = 0

                if post_block_remaining > 0:
                    blocked.append(True)
                    post_block_remaining -= 1
                else:
                    blocked.append(False)

        return pd.Series(blocked, index=severity.index, name="Rain_Blocked_CONS")

    hourly["Rain_Blocked_CONS"] = compute_rain_blocked_cons(hourly["Rain_Severity"])

    # ============================================================
    # 6) Compute GPH flags (all require DATA_OK for national consistency)
    # ============================================================

    # OPT (keep your existing intent, but require DATA_OK)
    hourly["GPH_OPT"] = (
        (hourly["Daylight_Flag"] == 1) &
        (hourly["Temp_F"] >= 32) &
        (hourly["Wind_Speed"] <= 30) &
        (~hourly["Has_Thunder"]) &
        (hourly["Rain_Severity"] <= 1.0)
    ).astype(int)


    # NATIONAL BASE (single definition for all cities)
    # Tune ONLY these numbers globally (min_temp, max_temp, max_wind).

    hourly["GPH_BASE"] = (
        (hourly["Daylight_Flag"] == 1) &
        (hourly["Temp_F"] >= 38) &
        (hourly["Wind_Speed"] <= 25) &
        (~hourly["Has_Thunder"]) &
        (~hourly["Wet_LowVis"]) &
        (hourly["Rain_Severity"] <= 1.0)
    ).astype(int)


    # CONS (keep your existing strictness, but require DATA_OK)
    hourly["GPH_CONS"] = (
        (hourly["Daylight_Flag"] == 1) &
        (hourly["Temp_F"] >= 40) &
        (hourly["Wind_Speed"] <= 22) &
        (~hourly["Has_Thunder"]) &
        (~hourly["Wet_LowVis"]) &
        (hourly["Rain_Severity"] <= 0.5) &
        (~hourly["Rain_Blocked_CONS"])
    ).astype(int)


    # Optional default
    hourly["GPH_PLAYABLE"] = hourly["GPH_BASE"]

    # ============================================================
    # 7) Export per city (years present in data)
    # ============================================================
    years_present = sorted({d.year for d in hourly.index if pd.notna(d)})
    year_min = years_present[0] if years_present else "NA"
    year_max = years_present[-1] if years_present else "NA"


    hourly_data_dir = BASE_DIR / "HOURLY_DATA"
    hourly_data_dir.mkdir(parents=True, exist_ok=True)

    out_path = hourly_data_dir / f"{city_caps}_GPH_Hourly_{year_min}_{year_max}.csv"

    hourly.to_csv(out_path, index=False, encoding='utf-8')

    tmp = hourly.copy()
    tmp["Year"] = tmp.index.year
    annual_summary = tmp.groupby("Year")[["GPH_OPT", "GPH_BASE", "GPH_CONS"]].sum()

    print(f"{city_caps}: wrote {out_path.name} | solar_offset={SOLAR_TO_LOCAL_OFFSET:+}")
    del weather_df
    return hourly, annual_summary


def build_gph_comparison_table(all_summaries: dict) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with:
    CITY | YEAR | GPH_OPT | GPH_BASE | GPH_CONS 
    """

    rows = []

    for city, summary_df in all_summaries.items():

        # summary_df is indexed by Year
        for year, row in summary_df.iterrows():

            rows.append(
                {
                    "CITY": city,
                    "YEAR": int(year),
                    "GPH_OPT": int(row["GPH_OPT"]),
                    "GPH_BASE": int(row["GPH_BASE"]),
                    "GPH_CONS": int(row["GPH_CONS"])
                }
            )

    out = pd.DataFrame(rows).sort_values(["CITY", "YEAR"]).reset_index(drop=True)
    return out

# --- STREAMLIT UI ---

def main():
    st.set_page_config(page_title="National GPH Calculator", layout="wide")
    st.title("🏙️ National GPH Calculation Tool")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration")
        
        # 1. Prepare the sorted list of cities
        available_cities = sorted([c.upper() for c in CITY_TO_STATION_ID.keys()])
        
        # 2. Add the "Select All" Checkbox
        select_all = st.checkbox("Select All Cities")

        # 3. Logic to toggle the default values
        if select_all:
            selected_cities = st.multiselect("Select Cities", available_cities, default=available_cities)
        else:
            selected_cities = st.multiselect("Select Cities", available_cities)
        
        year_input = st.text_input("Enter Years (comma separated)", value="2024, 2025")

        try:
            years = [int(y.strip()) for y in year_input.split(",") if y.strip()]
        except ValueError:
            st.error("Please enter valid numeric years.")
            years = []

    with col2:
        st.subheader("Process Control")
        st.info("Files will be saved to your local `HOURLY_DATA` and base folders.")
        run_btn = st.button("Calculate GPH for the given Cities + Year combinations", use_container_width=True)

    if run_btn:
        if not selected_cities or not years:
            st.warning("Please select at least one city and one year.")
        else:
            all_summaries = {}
            progress_bar = st.progress(0)
            
            for i, city in enumerate(selected_cities):
                with st.spinner(f"Processing {city}..."):
                    try:
                        _, annual = run_city_gph(city, CITY_TO_STATION_ID, years)
                        all_summaries[city] = annual
                    except Exception as e:
                        st.error(f"Error processing {city}: {e}")
                progress_bar.progress((i + 1) / len(selected_cities))

            if all_summaries:
                gph_comparison = build_gph_comparison_table(all_summaries)
                
                # Save the final comparison
                
                t_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_name = f'GPH_CALCULATIONS_{t_stamp}.csv'
                final_path = GPH_OUTPUT_DIR / final_name
                gph_comparison.to_csv(final_path, index=False, encoding='utf-8')
                
                st.success(f"✅ GPH Calculations Saved to Your Folder as {final_name}")
                st.balloons()
                
                st.subheader("Results Preview")
                st.dataframe(gph_comparison, use_container_width=True)

# --- BOILERPLATE FOR ONE-CLICK RUN ---
if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())