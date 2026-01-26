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

# Cloud-agnostic pathing: GitHub files are in the same directory as app.py
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

# --- HELPER FUNCTIONS ---

def fetch_noaa_lcd_cloud(station_id: str, year: int) -> pd.DataFrame:
    """Fetches LCD data directly from NOAA servers into memory."""
    url = f"https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/{year}/LCD_{station_id}_{year}.csv"
    try:
        # Streamlit cloud handles the SSL certificates automatically
        df = pd.read_csv(url, low_memory=False, encoding='utf-8')
        df["DATE"] = pd.to_datetime(df["DATE"])
        return df[[
            "DATE", "HourlyDryBulbTemperature", "HourlyWindSpeed", 
            "HourlyPrecipitation", "HourlyPresentWeatherType", "HourlyVisibility"
        ]]
    except Exception as e:
        st.error(f"Could not fetch NOAA data for {station_id} in {year}. (URL: {url})")
        return None

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
    except Exception:
        return np.nan

def load_sunrise_sunset_cloud(station_id: str) -> tuple[pd.DataFrame, dict]:
    """Loads Sunrise/Sunset files that you uploaded to your GitHub repository."""
    # This assumes you uploaded files named 'USW000..._SUNRISE_SUNSET.csv' to your GitHub
    csv_path = BASE_DIR / "SUNRISE_SUNSET" /f"{station_id}_SUNRISE_SUNSET.csv"
    
    if not csv_path.exists():
        st.error(f"Reference file missing on GitHub: {csv_path.name}")
        return None, None

    sun = pd.read_csv(csv_path, encoding='utf-8')
    sun.columns = [c.strip() for c in sun.columns]
    sun["Date"] = pd.to_datetime(sun["Date"], errors="coerce").dt.normalize()
    sun["sr_float"] = sun["SUNRISE_LST"].apply(time_to_float)
    sun["ss_float"] = sun["SUNSET_LST"].apply(time_to_float)
    
    sun_master = sun.dropna(subset=["Date", "sr_float", "ss_float"]).copy()
    sun_lookup = sun_master.set_index("Date")[["sr_float", "ss_float"]].to_dict("index")
    return sun_master, sun_lookup

def run_city_gph(city_caps: str, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_key = city_caps.title() if city_caps != "DC" else "DC"
    station_id = CITY_TO_STATION_ID[city_key]

    # --- 1. Load Weather Data from NOAA (Cloud) ---
    all_weather_dfs = []
    for yr in years:
        df_yr = fetch_noaa_lcd_cloud(station_id, yr)
        if df_yr is not None:
            all_weather_dfs.append(df_yr)
    
    if not all_weather_dfs:
        raise ValueError(f"No weather data found for {city_caps}")
    
    weather_df = pd.concat(all_weather_dfs, ignore_index=True)

    # --- 2. Load Sunrise Data from GitHub ---
    sun_master, sun_lookup = load_sunrise_sunset_cloud(station_id)
    if sun_lookup is None:
        raise FileNotFoundError(f"Sunrise data missing for {station_id}")

    # Force sunset to legacy 12-hour PM convention
    for d in sun_lookup:
        if sun_lookup[d]["ss_float"] > 12:
            sun_lookup[d]["ss_float"] -= 12

    # --- 3. Processing Logic (Original Formulas) ---
    weather_df["Hour_Rounded"] = weather_df["DATE"].dt.floor("H")
    hourly = weather_df.drop_duplicates("Hour_Rounded", keep="last").set_index("Hour_Rounded")
    hourly["DATA_OK"] = 1
    
    full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="H")
    hourly = hourly.reindex(full_idx)
    hourly["DATA_OK"] = hourly["DATA_OK"].fillna(0).astype(int)

    # Clean and Interpolate
    def clean_to_float(series):
        return pd.to_numeric(series.astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce")

    hourly["Temp_F"] = clean_to_float(hourly["HourlyDryBulbTemperature"]) * 9/5 + 32
    hourly["Temp_F"] = hourly["Temp_F"].interpolate(method="time").ffill().bfill()
    
    hourly["Wind_Speed"] = clean_to_float(hourly["HourlyWindSpeed"]).interpolate(method="time").ffill().bfill()
    hourly["Visibility_Mi"] = clean_to_float(hourly["HourlyVisibility"]).interpolate(method="time").ffill().bfill()
    
    hourly["Precip"] = clean_to_float(hourly["HourlyPrecipitation"].astype(str).str.strip().replace("T", "0.001"))
    hourly.loc[hourly["DATA_OK"] == 1, "Precip"] = hourly.loc[hourly["DATA_OK"] == 1, "Precip"].fillna(0.0)

    # Flags
# --- 3. Processing Logic (Updated for 'upper' error) ---
    
    # Ensure the column is treated as strings, then use .str.upper()
    wc = hourly["HourlyPresentWeatherType"].fillna("").astype(str).str.upper()
    
    # Use .str.contains for vectorized string searching
    hourly["Has_Thunder"] = wc.str.contains(r"TS|TSTM|LTG", regex=True)
    hourly["Wet_LowVis"] = (wc.str.contains(r"RA|DZ|BR|FG", regex=True)) & (hourly["Visibility_Mi"] < 6)

    # Update Rain_Severity to use vectorized logic
    hourly["Rain_Severity"] = 0.0
    hourly.loc[wc.str.contains(r"RA", na=False), "Rain_Severity"] = 1.0
    hourly.loc[wc.str.contains(r"\+RA", na=False), "Rain_Severity"] = 2.0

    # Daylight logic
    def get_daylight_flag(ts):
        dt = ts.normalize()
        if dt not in sun_lookup: return 0
        hr = ts.hour + ts.minute / 60.0
        sr, ss = sun_lookup[dt]["sr_float"] - 0.5, sun_lookup[dt]["ss_float"] + 0.5
        return 1 if (sr <= hr <= ss) else 0

    hourly["Daylight_Flag"] = [get_daylight_flag(ts) for ts in hourly.index]

    # Playability
    hourly["Rain_Severity"] = [0.0 if "RA" not in c else (2.0 if "+RA" in c else 1.0) for c in wc] # Simplified for brevity
    
    hourly["GPH_OPT"] = ((hourly["Daylight_Flag"] == 1) & (hourly["Temp_F"] >= 32) & (hourly["Wind_Speed"] <= 30) & (~hourly["Has_Thunder"])).astype(int)
    hourly["GPH_BASE"] = ((hourly["Daylight_Flag"] == 1) & (hourly["Temp_F"] >= 38) & (hourly["Wind_Speed"] <= 25) & (~hourly["Has_Thunder"])).astype(int)
    hourly["GPH_CONS"] = ((hourly["Daylight_Flag"] == 1) & (hourly["Temp_F"] >= 40) & (hourly["Wind_Speed"] <= 22) & (~hourly["Has_Thunder"])).astype(int)

    tmp = hourly.copy()
    tmp["Year"] = tmp.index.year
    annual_summary = tmp.groupby("Year")[["GPH_OPT", "GPH_BASE", "GPH_CONS"]].sum()

    return hourly, annual_summary

def main():
    st.set_page_config(page_title="Cloud GPH Calculator", layout="wide")
    st.title("🏙️ National GPH Calculation Tool (Cloud Edition)")
    st.info("Data is fetched live from NOAA. No local files required.")

    col1, col2 = st.columns(2)
    with col1:
        available_cities = sorted([c.upper() for c in CITY_TO_STATION_ID.keys()])
        select_all = st.checkbox("Select All Cities")
        selected_cities = st.multiselect("Select Cities", available_cities, default=available_cities if select_all else [])
        year_input = st.text_input("Enter Years (comma separated)", value="2024, 2025")

    if st.button("Calculate GPH", use_container_width=True):
        years = [int(y.strip()) for y in year_input.split(",") if y.strip()]
        all_summaries = {}
        progress = st.progress(0)
        
        for i, city in enumerate(selected_cities):
            with st.spinner(f"Fetching data for {city}..."):
                try:
                    _, annual = run_city_gph(city, years)
                    all_summaries[city] = annual
                except Exception as e:
                    st.error(f"Error {city}: {e}")
            progress.progress((i + 1) / len(selected_cities))

        if all_summaries:
            rows = []
            for city, summ in all_summaries.items():
                for yr, row in summ.iterrows():
                    rows.append({"CITY": city, "YEAR": yr, **row.to_dict()})
            
            final_df = pd.DataFrame(rows)
            st.success("Analysis Complete!")
            st.dataframe(final_df, use_container_width=True)
            
            # Allow the client to download the result since we don't save to their local disk
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name="GPH_Analysis.csv", mime="text/csv")

if __name__ == "__main__":
    main()
