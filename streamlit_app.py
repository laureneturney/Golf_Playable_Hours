"""
Advanced GPH Calculator - Cloud-Based Solution
Combines NOAA data fetching with local data support, station management, and GPH calculations
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import os
import math
from io import StringIO
from functools import lru_cache
import time

# --- CONFIGURATION & SETTINGS ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = Path.cwd()
SUN_DIR = BASE_DIR / "SUNRISE_SUNSET"
STATIONS_DB_PATH = BASE_DIR / "stations_db.json"

# Ensure sunrise/sunset cache directory exists
SUN_DIR.mkdir(parents=True, exist_ok=True)

# Default stations (can be expanded by user)
DEFAULT_CITY_TO_STATION_ID = {
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

# ============================================================================
# STATION DATABASE MANAGEMENT
# ============================================================================

class StationDatabase:
    """Manages persistent storage of user-defined stations"""
    
    def __init__(self, db_path: Path = STATIONS_DB_PATH):
        self.db_path = db_path
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load station database from JSON file"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return self._create_default()
        return self._create_default()
    
    def _create_default(self) -> Dict:
        """Create default database structure"""
        return {
            "stations": {name: {
                "station_id": sid,
                "station_name": name,
                "full_name": "",
                "latitude": "",
                "longitude": "",
                "climate_region": "",
                "utc_offset": "",
                "solar_time": "",
                "civil_time": "",
                "is_default": True,
                "date_added": datetime.now().isoformat()
            } for name, sid in DEFAULT_CITY_TO_STATION_ID.items()}
        }
    
    def save(self):
        """Save database to JSON file"""
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_station(self, station_name: str, station_id: str, metadata: Dict):
        """Add or update a station"""
        self.data["stations"][station_name] = {
            "station_id": station_id,
            "station_name": station_name,
            "full_name": metadata.get("full_name", ""),
            "latitude": metadata.get("latitude", ""),
            "longitude": metadata.get("longitude", ""),
            "climate_region": metadata.get("climate_region", ""),
            "utc_offset": metadata.get("utc_offset", ""),
            "solar_time": metadata.get("solar_time", ""),
            "civil_time": metadata.get("civil_time", ""),
            "is_default": metadata.get("is_default", False),
            "date_added": datetime.now().isoformat()
        }
        self.save()
    
    def get_all_stations(self) -> Dict:
        """Get all stations"""
        return self.data.get("stations", {})
    
    def get_station(self, station_name: str) -> Optional[Dict]:
        """Get a specific station"""
        return self.data.get("stations", {}).get(station_name)
    
    def rename_station(self, old_name: str, new_name: str) -> bool:
        """Rename a station's key (short name). Preserves all metadata."""
        stations = self.data.get("stations", {})
        if old_name not in stations or not new_name.strip():
            return False
        if new_name.strip() == old_name:
            return False
        if new_name.strip() in stations:
            return False  # name already taken
        station_data = stations.pop(old_name)
        station_data["station_name"] = new_name.strip()
        stations[new_name.strip()] = station_data
        self.save()
        return True

    def delete_station(self, station_name: str):
        """Delete a custom station (not default)"""
        station = self.get_station(station_name)
        if station and not station.get("is_default", False):
            del self.data["stations"][station_name]
            self.save()
            return True
        return False


# ============================================================================
# NOAA DATA FETCHING & STATION SEARCH
# ============================================================================

def geocode_location(query: str) -> Optional[Tuple[float, float]]:
    """
    Convert a location name or coordinate string to (lat, lon).

    Supports:
      - Coordinate input: "40.7,-74.0" -> (40.7, -74.0)
      - Location name: "New York" -> geocoded via Nominatim

    Returns:
        (latitude, longitude) tuple, or None on failure
    """
    query = query.strip()

    # Try to parse as direct coordinates
    if "," in query:
        parts = query.split(",")
        if len(parts) == 2:
            try:
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
            except ValueError:
                pass

    # Geocode using Nominatim (OpenStreetMap) - free, no API key
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "limit": 1,
            "countrycodes": "us"
        }
        headers = {"User-Agent": "GPH-Calculator/1.0 (streamlit-app)"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        results = resp.json()

        if results and len(results) > 0:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            return (lat, lon)
        return None
    except Exception:
        return None


def search_noaa_stations(lat: float, lon: float, limit: int = 25) -> List[Dict]:
    """
    Search NOAA NCEI for LCD stations near the given coordinates.

    Uses the NCEI Search Service v1 with a bounding box query on the
    local-climatological-data-v2 dataset. Results are deduplicated by
    station ID since the API returns one result per file (station+year).

    Args:
        lat: Latitude of search center
        lon: Longitude of search center
        limit: Maximum number of API results to fetch (before dedup)

    Returns:
        List of dicts with keys: name, station_id, latitude, longitude
        Sorted alphabetically by station name. Empty list on failure.
    """
    try:
        # Build bounding box: ~0.5 degrees (~35 miles) in each direction
        bbox_half = 0.5
        north = min(lat + bbox_half, 90.0)
        south = max(lat - bbox_half, -90.0)
        east = min(lon + bbox_half, 180.0)
        west = max(lon - bbox_half, -180.0)

        # NCEI bbox format: N,W,S,E
        bbox_str = f"{north},{west},{south},{east}"

        url = "https://www.ncei.noaa.gov/access/services/search/v1/data"
        params = {
            "dataset": "local-climatological-data-v2",
            "bbox": bbox_str,
            "startDate": "2023-01-01T00:00:00",
            "endDate": "2025-12-31T23:59:59",
            "limit": limit,
            "offset": 0
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Deduplicate by station ID
        seen_ids = {}
        for result in data.get("results", []):
            stations = result.get("stations", [])
            location = result.get("location", {})
            coords = location.get("coordinates", [None, None])

            for station in stations:
                sid = station.get("id", "")
                sname = station.get("name", "")

                if sid and sid not in seen_ids:
                    seen_ids[sid] = {
                        "name": sname,
                        "station_id": sid,
                        "latitude": str(coords[1]) if coords[1] is not None else "",
                        "longitude": str(coords[0]) if coords[0] is not None else "",
                    }

        return sorted(seen_ids.values(), key=lambda x: x["name"])

    except Exception:
        return []


def build_station_display_name(station: Dict) -> str:
    """Build a display string like 'NEWARK LIBERTY INTL AIRPORT, NJ US (USW00014734)'"""
    name = station.get("name", "Unknown")
    sid = station.get("station_id", "")
    return f"{name} ({sid})"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_noaa_lcd_data(station_id: str, year: int) -> Optional[pd.DataFrame]:
    """
    Fetch LCD data from NOAA cloud storage.

    Args:
        station_id: NOAA station ID (e.g., "USW00094846")
        year: Year to fetch

    Returns:
        DataFrame with weather data or None if fetch fails
    """
    url = f"https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/{year}/LCD_{station_id}_{year}.csv"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), low_memory=False)
        df["DATE"] = pd.to_datetime(df["DATE"])
        needed = [
            "DATE",
            "HourlyDryBulbTemperature",
            "HourlyWindSpeed",
            "HourlyPrecipitation",
            "HourlyPresentWeatherType",
            "HourlyVisibility"
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return None
        return df[needed]
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def extract_station_metadata_from_lcd(station_id: str, year: int = 2024) -> Optional[Dict]:
    """
    Extract station metadata from NOAA cloud LCD data.

    Args:
        station_id: NOAA station ID
        year: Year to fetch a sample file from

    Returns:
        Dictionary with metadata or None
    """
    url = f"https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/{year}/LCD_{station_id}_{year}.csv"
    try:
        resp = requests.get(url, timeout=30, headers={"Range": "bytes=0-8192"})
        if resp.status_code not in (200, 206):
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
        text = resp.text
        # Read just the first few rows
        df = pd.read_csv(StringIO(text), nrows=2, low_memory=False)

        lat = str(df["LATITUDE"].iloc[0]) if "LATITUDE" in df.columns else ""
        lon = str(df["LONGITUDE"].iloc[0]) if "LONGITUDE" in df.columns else ""
        name = str(df["NAME"].iloc[0]) if "NAME" in df.columns else "Unknown"
        utc_offset = get_station_utc_offset(name, float(lon) if lon else 0.0)

        return {
            "full_name": name,
            "latitude": lat,
            "longitude": lon,
            "climate_region": "",
            "utc_offset": str(utc_offset),
            "solar_time": str(utc_offset),
            "civil_time": "",
        }
    except Exception:
        return None


# ============================================================================
# US STATE -> STANDARD TIME UTC OFFSET
# ============================================================================

_STATE_TO_UTC_OFFSET = {
    "CT": -5, "DC": -5, "DE": -5, "FL": -5, "GA": -5, "IN": -5,
    "KY": -5, "MA": -5, "MD": -5, "ME": -5, "MI": -5, "NC": -5,
    "NH": -5, "NJ": -5, "NY": -5, "OH": -5, "PA": -5, "RI": -5,
    "SC": -5, "VA": -5, "VT": -5, "WV": -5,
    "AL": -6, "AR": -6, "IA": -6, "IL": -6, "KS": -6, "LA": -6,
    "MN": -6, "MO": -6, "MS": -6, "ND": -6, "NE": -6, "OK": -6,
    "SD": -6, "TN": -6, "TX": -6, "WI": -6,
    "AZ": -7, "CO": -7, "ID": -7, "MT": -7, "NM": -7, "UT": -7,
    "WY": -7,
    "CA": -8, "NV": -8, "OR": -8, "WA": -8,
    "AK": -9, "HI": -10,
}


def get_station_utc_offset(station_name: str, longitude: float = 0.0) -> int:
    """
    Determine the standard-time UTC offset for a US weather station.

    Extracts the 2-letter state code from the NOAA station name
    (format: 'STATION NAME, STATE US') and maps to UTC offset.
    Falls back to longitude-based estimate.
    """
    # Try to extract state from "CITY NAME, ST US" pattern
    name = station_name.strip().upper()
    if " US" in name:
        before_us = name.split(" US")[0].strip()
        if ", " in before_us:
            state = before_us.rsplit(", ", 1)[-1].strip()
            if len(state) == 2 and state in _STATE_TO_UTC_OFFSET:
                return _STATE_TO_UTC_OFFSET[state]
        # Try 2-letter suffix before " US"
        parts = before_us.split()
        if parts:
            candidate = parts[-1].strip()
            if len(candidate) == 2 and candidate in _STATE_TO_UTC_OFFSET:
                return _STATE_TO_UTC_OFFSET[candidate]

    # Fallback: estimate from longitude
    if longitude != 0.0:
        return round(longitude / 15)
    return -6  # Default to Central US


# ============================================================================
# NOAA SOLAR CALCULATOR - Sunrise / Sunset Computation
# ============================================================================

def _noaa_sunrise_sunset(date_obj, lat: float, lon: float, utc_offset: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute sunrise and sunset times for a single date using the
    NOAA Solar Calculator algorithm (based on Jean Meeus,
    'Astronomical Algorithms').

    Args:
        date_obj: datetime.date or datetime.datetime
        lat: Latitude in degrees (+ North)
        lon: Longitude in degrees (+ East, US is negative)
        utc_offset: Standard-time UTC offset (e.g., -5 for EST)

    Returns:
        (sunrise_hours, sunset_hours) as floats in local standard time,
        or (None, None) for polar day/night.
    """
    import datetime as _dt
    if isinstance(date_obj, _dt.datetime):
        date_obj = date_obj.date()

    # Julian Day Number
    y = date_obj.year
    m = date_obj.month
    d = date_obj.day
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5

    # Julian Century
    jc = (jd - 2451545.0) / 36525.0

    # Sun geometry
    geom_mean_long = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    geom_mean_anom = (357.52911 + jc * (35999.05029 - jc * 0.0001537)) % 360
    ecc_earth = 0.016708634 - jc * (0.000042037 + jc * 0.0000001267)

    anom_rad = math.radians(geom_mean_anom)
    eq_center = (math.sin(anom_rad) * (1.914602 - jc * (0.004817 + jc * 0.000014))
                 + math.sin(2 * anom_rad) * (0.019993 - jc * 0.000101)
                 + math.sin(3 * anom_rad) * 0.000289)

    sun_true_long = geom_mean_long + eq_center
    sun_app_long = sun_true_long - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * jc))

    # Obliquity
    mean_obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq_corr = mean_obliq + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * jc))

    # Declination
    sun_decl = math.degrees(math.asin(
        math.sin(math.radians(obliq_corr)) * math.sin(math.radians(sun_app_long))
    ))

    # Equation of Time (minutes)
    obliq_rad = math.radians(obliq_corr)
    y_var = math.tan(obliq_rad / 2) ** 2
    l0_rad = math.radians(geom_mean_long)
    eq_time = 4 * math.degrees(
        y_var * math.sin(2 * l0_rad)
        - 2 * ecc_earth * math.sin(anom_rad)
        + 4 * ecc_earth * y_var * math.sin(anom_rad) * math.cos(2 * l0_rad)
        - 0.5 * y_var ** 2 * math.sin(4 * l0_rad)
        - 1.25 * ecc_earth ** 2 * math.sin(2 * anom_rad)
    )

    # Hour angle at sunrise/sunset (solar zenith = 90.833 degrees)
    lat_rad = math.radians(lat)
    decl_rad = math.radians(sun_decl)
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(lat_rad) * math.cos(decl_rad))
              - math.tan(lat_rad) * math.tan(decl_rad))

    # Polar day/night check
    if cos_ha < -1 or cos_ha > 1:
        return None, None

    ha = math.degrees(math.acos(cos_ha))

    # Solar noon (minutes from midnight, local standard time)
    solar_noon = 720 - 4 * lon - eq_time + utc_offset * 60

    # Sunrise and sunset in minutes from midnight LST
    sunrise_min = solar_noon - ha * 4
    sunset_min = solar_noon + ha * 4

    return sunrise_min / 60.0, sunset_min / 60.0


def compute_sunrise_sunset(lat: float, lon: float, utc_offset: int,
                           start_year: int = 2021, end_year: int = 2051) -> pd.DataFrame:
    """
    Compute daily sunrise/sunset for a location across multiple years.
    Produces the same format as the pre-generated SUNRISE_SUNSET CSV files.

    Returns:
        DataFrame with columns: Date, SUNRISE_LST, SUNSET_LST
    """
    import datetime as _dt
    rows = []
    current = _dt.date(start_year, 1, 1)
    end = _dt.date(end_year, 9, 15)

    while current <= end:
        sr, ss = _noaa_sunrise_sunset(current, lat, lon, utc_offset)
        if sr is not None and ss is not None:
            sr_h = int(sr)
            sr_m = int((sr - sr_h) * 60)
            sr_s = int(((sr - sr_h) * 60 - sr_m) * 60)
            ss_h = int(ss)
            ss_m = int((ss - ss_h) * 60)
            ss_s = int(((ss - ss_h) * 60 - ss_m) * 60)
            rows.append({
                "Date": current.strftime("%-m/%-d/%Y"),
                "SUNRISE_LST": f"{sr_h}:{sr_m:02d}:{sr_s:02d}",
                "SUNSET_LST": f"{ss_h}:{ss_m:02d}:{ss_s:02d}",
            })
        current += _dt.timedelta(days=1)

    return pd.DataFrame(rows)


# ============================================================================
# SUNRISE / SUNSET LOADING (auto-compute if file missing)
# ============================================================================

def load_sunrise_sunset_data(
    station_id: str,
    latitude: Optional[str] = None,
    longitude: Optional[str] = None,
    utc_offset: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Load sunrise/sunset data for a station.
    If a pre-generated CSV exists, use it.  Otherwise compute from
    lat/lon/utc_offset using the NOAA solar algorithm and save the
    result for future reuse.

    Args:
        station_id: NOAA station ID
        latitude: Station latitude (string, needed if computing)
        longitude: Station longitude (string, needed if computing)
        utc_offset: UTC offset string (e.g., "-5", needed if computing)

    Returns:
        Tuple of (dataframe, lookup dict)
    """
    csv_path = SUN_DIR / f"{station_id}_SUNRISE_SUNSET.csv"

    # If file doesn't exist, try to compute it
    if not csv_path.exists():
        if latitude and longitude:
            try:
                lat = float(latitude)
                lon = float(longitude)
                tz = int(float(utc_offset)) if utc_offset else get_station_utc_offset("", lon)
                sun_df = compute_sunrise_sunset(lat, lon, tz)
                SUN_DIR.mkdir(parents=True, exist_ok=True)
                sun_df.to_csv(csv_path, index=False)
            except Exception:
                return None, None
        else:
            return None, None

    try:
        sun = pd.read_csv(csv_path, encoding='utf-8')
        sun.columns = [c.strip() for c in sun.columns]
        sun["Date"] = pd.to_datetime(sun["Date"], errors="coerce").dt.normalize()
        sun["sr_float"] = sun["SUNRISE_LST"].apply(_time_to_float)
        sun["ss_float"] = sun["SUNSET_LST"].apply(_time_to_float)

        # Restore legacy 12-hour PM convention
        sun["ss_float"] = sun["ss_float"].where(
            sun["ss_float"] <= 12,
            sun["ss_float"] - 12
        )

        sun_lookup = sun.dropna(subset=["Date", "sr_float", "ss_float"]).set_index("Date")[
            ["sr_float", "ss_float"]
        ].to_dict("index")

        return sun, sun_lookup
    except Exception:
        return None, None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _time_to_float(t):
    """Convert time string to float hours"""
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


def _clean_to_float(series: pd.Series) -> pd.Series:
    """Clean and convert series to float"""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.-]", "", regex=True),
        errors="coerce"
    )


def _interpolate_all_time(s: pd.Series, fill_edges: bool = True) -> pd.Series:
    """Interpolate series across time"""
    s = s.copy()
    s = s.interpolate(method="time", limit_direction="both" if fill_edges else "forward")
    if fill_edges:
        s = s.ffill().bfill()
    return s


def _classify_rain(code: str, precip: float) -> float:
    """Classify rain severity from weather code and precipitation"""
    code = (code or "").upper()
    p = float(precip) if pd.notna(precip) else np.nan
    
    if pd.isna(p) and code == "":
        return np.nan
    
    if ("+RA" in code) or ("+TS" in code) or (pd.notna(p) and p >= 0.25):
        return 2.0
    if (("RA" in code) and ("-RA" not in code)) or (pd.notna(p) and p >= 0.10):
        return 1.0
    if ("-RA" in code) or ("DZ" in code) or (pd.notna(p) and p > 0):
        return 0.5
    return 0.0


def _compute_rain_blocked_cons(severity: pd.Series) -> pd.Series:
    """Compute rain blocking for conservative GPH"""
    blocked = []
    heavy_streak = 0
    post_block_remaining = 0
    
    for sev in severity.values:
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
    
    return pd.Series(blocked, index=severity.index)


# ============================================================================
# CORE GPH CALCULATION ENGINE
# ============================================================================

def calculate_gph(
    station_id: str,
    station_name: str,
    years: List[int],
    progress_callback=None,
    latitude: Optional[str] = None,
    longitude: Optional[str] = None,
    utc_offset: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Calculate GPH for a station across multiple years

    Args:
        station_id: NOAA station ID
        station_name: User-friendly station name
        years: List of years to process
        progress_callback: Optional callback for progress updates
        latitude: Station latitude (for auto-generating sunrise/sunset)
        longitude: Station longitude (for auto-generating sunrise/sunset)
        utc_offset: UTC offset string (for auto-generating sunrise/sunset)

    Returns:
        Tuple of (hourly dataframe, annual summary)
    """

    try:
        # 1. Fetch weather data for all years
        weather_dfs = []
        for year in years:
            if progress_callback:
                progress_callback(f"Fetching data for {station_name} - {year}...")

            df = fetch_noaa_lcd_data(station_id, year)
            if df is not None:
                weather_dfs.append(df)

        if not weather_dfs:
            return None, None

        weather_df = pd.concat(weather_dfs, ignore_index=True)

        # 2. If we don't have lat/lon/offset yet, try to extract from LCD data
        if not latitude or not longitude:
            meta = extract_station_metadata_from_lcd(station_id, years[0])
            if meta:
                latitude = latitude or meta.get("latitude", "")
                longitude = longitude or meta.get("longitude", "")
                utc_offset = utc_offset or meta.get("utc_offset", "")

        # 3. Load sunrise/sunset data (auto-computes if file missing)
        _, sun_lookup = load_sunrise_sunset_data(
            station_id, latitude=latitude, longitude=longitude, utc_offset=utc_offset
        )
        if sun_lookup is None:
            return None, None
        
        # 3. Create hourly index with DATA_OK flag
        weather_df["Hour_Rounded"] = weather_df["DATE"].dt.floor("H")
        hourly = weather_df.drop_duplicates("Hour_Rounded", keep="last").set_index("Hour_Rounded")
        hourly["DATA_OK"] = 1
        
        full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="H")
        hourly = hourly.reindex(full_idx)
        hourly["DATA_OK"] = hourly["DATA_OK"].fillna(0).astype(int)
        
        # 4. Clean and interpolate continuous variables
        hourly["Temp_F"] = _interpolate_all_time(
            _clean_to_float(hourly["HourlyDryBulbTemperature"]) * 9/5 + 32,
            fill_edges=True
        )
        
        hourly["Wind_Speed"] = _interpolate_all_time(
            _clean_to_float(hourly["HourlyWindSpeed"]),
            fill_edges=True
        )
        
        hourly["Visibility_Mi"] = _interpolate_all_time(
            _clean_to_float(hourly["HourlyVisibility"]),
            fill_edges=True
        )
        
        # Precipitation - don't interpolate, just clean
        hourly["Precip_raw"] = hourly["HourlyPrecipitation"].astype(str).str.strip().replace("T", "0.001")
        hourly["Precip"] = _clean_to_float(hourly["Precip_raw"])
        hourly.loc[hourly["DATA_OK"] == 1, "Precip"] = hourly.loc[
            hourly["DATA_OK"] == 1, "Precip"
        ].fillna(0.0)
        
        # 5. Weather codes and flags
        hourly["Weather_Code"] = hourly["HourlyPresentWeatherType"].fillna("")
        wc = hourly["Weather_Code"].astype(str).str.upper()
        
        hourly["Has_Thunder"] = wc.str.contains(
            r"\bTS\b|\bTSTM\b|\bTSRA\b|LTG", regex=True
        )
        hourly["Has_Wet_Code"] = wc.str.contains(
            r"\bRA\b|\bDZ\b|\bBR\b|\bFG\b", regex=True
        )
        hourly["Wet_LowVis"] = (hourly["Has_Wet_Code"]) & (hourly["Visibility_Mi"] < 6)
        
        # 6. Daylight flag
        def wrap24(x):
            return x % 24
        
        def get_daylight_flag(ts):
            dt = ts.normalize()
            if dt not in sun_lookup:
                return 0
            
            hr = ts.hour + ts.minute / 60.0
            sr = wrap24(sun_lookup[dt]["sr_float"]) - 0.5
            ss = wrap24(sun_lookup[dt]["ss_float"]) + 0.5
            
            # Handle sunrise/sunset with midnight wrap-around
            if sr <= ss:
                return 1 if (sr <= hr <= ss) else 0
            else:
                # Sunset after midnight (e.g., sunset at 5:40 PM becomes 5.67)
                return 1 if (hr >= sr or hr <= ss) else 0
        
        hourly["Daylight_Flag"] = [get_daylight_flag(ts) for ts in hourly.index]
        
        # 7. Rain severity and blocking
        hourly["Rain_Severity"] = [
            _classify_rain(code, precip)
            for code, precip in zip(hourly["Weather_Code"].astype(str), hourly["Precip"])
        ]
        hourly["Rain_Blocked_CONS"] = _compute_rain_blocked_cons(hourly["Rain_Severity"])
        
        # 8. GPH Formulas (match app.py exactly - NO DATA_OK in formulas)
        hourly["GPH_OPT"] = (
            (hourly["Daylight_Flag"] == 1) &
            (hourly["Temp_F"] >= 32) &
            (hourly["Wind_Speed"] <= 30) &
            (~hourly["Has_Thunder"]) &
            (hourly["Rain_Severity"] <= 1.0)
        ).astype(int)
        
        hourly["GPH_BASE"] = (
            (hourly["Daylight_Flag"] == 1) &
            (hourly["Temp_F"] >= 38) &
            (hourly["Wind_Speed"] <= 25) &
            (~hourly["Has_Thunder"]) &
            (~hourly["Wet_LowVis"]) &
            (hourly["Rain_Severity"] <= 1.0)
        ).astype(int)
        
        hourly["GPH_CONS"] = (
            (hourly["Daylight_Flag"] == 1) &
            (hourly["Temp_F"] >= 40) &
            (hourly["Wind_Speed"] <= 22) &
            (~hourly["Has_Thunder"]) &
            (~hourly["Wet_LowVis"]) &
            (hourly["Rain_Severity"] <= 0.5) &
            (~hourly["Rain_Blocked_CONS"])
        ).astype(int)
        
        # 9. Generate summary
        tmp = hourly.copy()
        tmp["Year"] = tmp.index.year
        summary = tmp.groupby("Year")[["GPH_OPT", "GPH_BASE", "GPH_CONS"]].sum()
        
        return hourly, summary
    
    except Exception as e:
        st.error(f"Error calculating GPH for {station_name}: {e}")
        return None, None


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(page_title="GPH Calculator", layout="wide")

    # Custom CSS for Golf Club Benchmarks branding
    st.markdown("""
    <style>
    /* --- Global --- */
    .stApp {
        background-color: #ffffff;
    }

    /* --- Header bar --- */
    header[data-testid="stHeader"] {
        background-color: #1b4332;
    }

    /* --- Sidebar collapse/expand arrows --- */
    button[data-testid="stBaseButton-headerNoPadding"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background-color: #f0f4f1;
        border-right: 2px solid #2d6a4f;
    }
    section[data-testid="stSidebar"] * {
        color: #1a1a2e !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h5 {
        color: #1b4332 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio label span,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #1a1a2e !important;
    }

    /* --- Primary buttons --- */
    .stButton > button[kind="primary"],
    .stButton > button {
        background-color: #2d6a4f;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #1b4332;
        color: #ffffff;
    }
    .stDownloadButton > button {
        background-color: #2d6a4f;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background-color: #1b4332;
        color: #ffffff;
    }

    /* --- Headings --- */
    h1 {
        color: #1b4332 !important;
        font-weight: 700 !important;
    }
    h2, h3, h4 {
        color: #2d6a4f !important;
    }

    /* --- Accent gold bar under title --- */
    .brand-accent {
        height: 4px;
        background: linear-gradient(90deg, #ffab00 0%, #2d6a4f 100%);
        border-radius: 2px;
        margin-bottom: 1rem;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab"] {
        color: #2d6a4f;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #2d6a4f !important;
        color: #1b4332 !important;
    }

    /* --- Dataframe --- */
    .stDataFrame {
        border: 1px solid #d8e4dc;
        border-radius: 6px;
    }

    /* --- Checkbox --- */
    .stCheckbox label span {
        color: #1a1a2e;
    }

    /* --- Multiselect tags --- */
    span[data-baseweb="tag"] {
        background-color: #2d6a4f !important;
    }

    /* --- Success / info boxes --- */
    .stSuccess {
        border-left-color: #2d6a4f !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Greenlight Advisors Golf Playable Hour (GPH) Calculator")
    st.markdown('<div class="brand-accent"></div>', unsafe_allow_html=True)
    st.markdown("Cloud-based solution with NOAA integration and station management")
    
    # Initialize session state
    if "db" not in st.session_state:
        st.session_state.db = StationDatabase()
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "search_error" not in st.session_state:
        st.session_state.search_error = ""

    # Sidebar for station management
    with st.sidebar:
        st.subheader("Station Management")
        
        tab1, tab2 = st.tabs(["View Stations", "Add Station"])
        
        with tab1:
            st.markdown("### Current Stations")
            stations = st.session_state.db.get_all_stations()

            station_list = []
            for name, data in stations.items():
                full = data.get("full_name", name)
                if data.get("is_default"):
                    label = full.replace("International Airport", "Intl").replace("International", "Intl")
                else:
                    label = full
                station_list.append((name, label))

            for key, label in sorted(station_list, key=lambda x: x[0]):
                st.write(f"• **{key}** — {label}")

            st.markdown(f"**Total**: {len(stations)} stations")

            st.divider()
            st.markdown("##### Edit Existing Stations")
            rename_target = st.selectbox(
                "Station to rename",
                options=sorted(stations.keys()),
                key="rename_station_select",
                label_visibility="collapsed",
            )
            new_name = st.text_input(
                "New short name",
                value=rename_target if rename_target else "",
                key="rename_station_input",
            )
            if st.button("Rename", use_container_width=True, key="rename_btn"):
                if rename_target and new_name.strip() and new_name.strip() != rename_target:
                    if new_name.strip() in stations:
                        st.warning("That name is already in use.")
                    else:
                        st.session_state.db.rename_station(rename_target, new_name.strip())
                        st.success(f"Renamed '{rename_target}' to '{new_name.strip()}'")
                        st.rerun()
        
        with tab2:
            st.markdown("### Add New Station")

            search_mode = st.radio(
                "Entry mode",
                ["Search NOAA", "Manual entry"],
                horizontal=True,
                label_visibility="collapsed"
            )

            if search_mode == "Search NOAA":
                st.caption("Search by city/state or coordinates (e.g., '40.7,-74.0')")

                search_col1, search_col2 = st.columns([3, 1])
                with search_col1:
                    search_query = st.text_input(
                        "Location",
                        placeholder="e.g., Newark NJ, or 40.7,-74.0",
                        key="station_search_input"
                    )
                with search_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    search_clicked = st.button("Search", use_container_width=True)

                if search_clicked and search_query.strip():
                    st.session_state.search_error = ""
                    st.session_state.search_results = []

                    with st.spinner("Searching NOAA stations..."):
                        coords = geocode_location(search_query.strip())

                        if coords is None:
                            st.session_state.search_error = (
                                f"Could not find location '{search_query}'. "
                                "Try a more specific name (e.g., 'Newark, NJ') "
                                "or enter coordinates directly (e.g., '40.7,-74.0')."
                            )
                        else:
                            lat, lon = coords
                            time.sleep(1)  # Nominatim rate limit
                            results = search_noaa_stations(lat, lon)

                            if results:
                                st.session_state.search_results = results
                                st.session_state.search_query = search_query.strip()
                            else:
                                st.session_state.search_error = (
                                    f"No LCD stations found near '{search_query}' "
                                    f"(searched around {lat:.2f}, {lon:.2f}). "
                                    "Try a different location or use manual entry."
                                )

                if st.session_state.search_error:
                    st.warning(st.session_state.search_error)

                if st.session_state.search_results:
                    station_options = [
                        build_station_display_name(s)
                        for s in st.session_state.search_results
                    ]

                    selected_display = st.selectbox(
                        "Select a station",
                        options=station_options,
                        key="station_dropdown"
                    )

                    selected_idx = station_options.index(selected_display)
                    selected_station = st.session_state.search_results[selected_idx]

                    # Sync short name when dropdown selection changes
                    suggested_name = selected_station["name"].title().split(",")[0].strip()
                    if st.session_state.get("_last_selected_station") != selected_station["station_id"]:
                        st.session_state["_last_selected_station"] = selected_station["station_id"]
                        st.session_state["custom_station_name"] = suggested_name

                    st.markdown("**Station details:**")
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.text(f"ID: {selected_station['station_id']}")
                        st.text(f"Lat: {selected_station['latitude']}")
                    with detail_col2:
                        st.text(f"Lon: {selected_station['longitude']}")

                    custom_name = st.text_input(
                        "Short name (used as station key)",
                        key="custom_station_name"
                    )

                    if st.button("Add Station", use_container_width=True, key="add_searched"):
                        if custom_name.strip():
                            with st.spinner("Fetching metadata & generating sunrise/sunset..."):
                                lat_str = selected_station["latitude"]
                                lon_str = selected_station["longitude"]
                                lon_f = float(lon_str) if lon_str else 0.0
                                tz = get_station_utc_offset(selected_station["name"], lon_f)

                                # Try to enrich metadata from LCD data
                                lcd_meta = extract_station_metadata_from_lcd(
                                    selected_station["station_id"]
                                )
                                if lcd_meta:
                                    lat_str = lat_str or lcd_meta.get("latitude", "")
                                    lon_str = lon_str or lcd_meta.get("longitude", "")

                                metadata = {
                                    "full_name": selected_station["name"],
                                    "latitude": lat_str,
                                    "longitude": lon_str,
                                    "climate_region": "",
                                    "utc_offset": str(tz),
                                    "solar_time": str(tz),
                                    "civil_time": "",
                                    "is_default": False
                                }

                                # Pre-generate sunrise/sunset file
                                if lat_str and lon_str:
                                    load_sunrise_sunset_data(
                                        selected_station["station_id"],
                                        latitude=lat_str,
                                        longitude=lon_str,
                                        utc_offset=str(tz),
                                    )

                            st.session_state.db.add_station(
                                custom_name.strip(),
                                selected_station["station_id"],
                                metadata
                            )
                            st.success(f"Station '{custom_name.strip()}' added!")
                            st.session_state.search_results = []
                            st.session_state.search_error = ""
                            st.rerun()
                        else:
                            st.warning("Please enter a short name for the station.")

            else:
                st.info(
                    "Visit NOAA to find station IDs: "
                    "https://www.ncei.noaa.gov/access/search/data-search/local-climatological-data-v2"
                )

                new_station_name = st.text_input("Station Name (e.g., 'My City')")
                new_station_id = st.text_input("Station ID (e.g., 'USW00094846')")
                new_full_name = st.text_input("Full Station Name (optional)")

                if st.button("Add Station", use_container_width=True, key="add_manual"):
                    if new_station_name and new_station_id:
                        with st.spinner("Fetching metadata from NOAA..."):
                            lcd_meta = extract_station_metadata_from_lcd(new_station_id)
                        lat = lcd_meta.get("latitude", "") if lcd_meta else ""
                        lon = lcd_meta.get("longitude", "") if lcd_meta else ""
                        full = lcd_meta.get("full_name", new_full_name or new_station_name) if lcd_meta else (new_full_name or new_station_name)
                        tz = lcd_meta.get("utc_offset", "") if lcd_meta else ""

                        # Pre-generate sunrise/sunset if we got coordinates
                        if lat and lon and tz:
                            load_sunrise_sunset_data(new_station_id, latitude=lat, longitude=lon, utc_offset=tz)

                        metadata = {
                            "full_name": full,
                            "latitude": lat,
                            "longitude": lon,
                            "climate_region": "",
                            "utc_offset": tz,
                            "solar_time": tz,
                            "civil_time": "",
                            "is_default": False
                        }
                        st.session_state.db.add_station(
                            new_station_name, new_station_id, metadata
                        )
                        st.success(f"Station '{new_station_name}' added!")
                        st.rerun()
                    else:
                        st.warning("Please fill in station name and ID")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Calculate GPH")
        
        stations = st.session_state.db.get_all_stations()
        station_names = sorted(stations.keys())

        select_all = st.checkbox("Select All Stations")

        selected_stations = st.multiselect(
            "Select Stations",
            station_names,
            default=station_names if select_all else (station_names[:3] if len(station_names) >= 3 else station_names),
        )
        
        year_input = st.text_input(
            "Years (comma-separated)",
            value="2024, 2025"
        )
    
    with col2:
        st.subheader("Options")
        export_format = st.radio("Export Format", ["CSV", "Excel"], horizontal=True)
    
    # Parse years
    try:
        years = [int(y.strip()) for y in year_input.split(",") if y.strip()]
    except ValueError:
        st.error("Please enter valid numeric years")
        years = []
    
    # Calculate GPH
    if st.button("Calculate GPH", use_container_width=True):
        if not selected_stations or not years:
            st.warning("Please select stations and years")
        else:
            progress_placeholder = st.empty()
            results = []
            monthly_results = []

            total_tasks = len(selected_stations)
            
            for idx, station_name in enumerate(selected_stations):
                station_data = stations[station_name]
                station_id = station_data["station_id"]
                
                with progress_placeholder.container():
                    st.info(f"Processing {station_name} ({idx+1}/{total_tasks})...")
                
                def progress_cb(msg):
                    with progress_placeholder.container():
                        st.info(f"{msg}")
                
                hourly, summary = calculate_gph(
                    station_id, station_name, years, progress_cb,
                    latitude=station_data.get("latitude"),
                    longitude=station_data.get("longitude"),
                    utc_offset=station_data.get("utc_offset") or station_data.get("solar_time"),
                )
                
                if summary is not None:
                    for year, row in summary.iterrows():
                        results.append({
                            "STATION": station_name,
                            "YEAR": int(year),
                            "GPH_OPT": int(row["GPH_OPT"]),
                            "GPH_BASE": int(row["GPH_BASE"]),
                            "GPH_CONS": int(row["GPH_CONS"])
                        })

                    # Collect monthly aggregates from hourly data
                    tmp = hourly[["GPH_OPT", "GPH_BASE", "GPH_CONS"]].copy()
                    tmp["YEAR"] = tmp.index.year
                    tmp["MONTH"] = tmp.index.month
                    monthly = tmp.groupby(["YEAR", "MONTH"])[
                        ["GPH_OPT", "GPH_BASE", "GPH_CONS"]
                    ].sum().reset_index()
                    monthly.insert(0, "STATION", station_name)
                    monthly_results.append(monthly)

            progress_placeholder.empty()

            if results:
                # --- Annual Results ---
                results_df = pd.DataFrame(results).sort_values(["STATION", "YEAR"])

                st.success("Calculation Complete!")
                st.subheader("Annual GPH Summary")
                st.dataframe(results_df, use_container_width=True)

                # Annual export
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                if export_format == "CSV":
                    st.download_button(
                        "Download Annual CSV",
                        data=results_df.to_csv(index=False),
                        file_name=f"GPH_Annual_{ts}.csv",
                        mime="text/csv",
                        key="dl_annual_csv",
                    )
                else:
                    from io import BytesIO
                    buf = BytesIO()
                    results_df.to_excel(buf, index=False, engine="openpyxl")
                    st.download_button(
                        "Download Annual Excel",
                        data=buf.getvalue(),
                        file_name=f"GPH_Annual_{ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_annual_xlsx",
                    )

                # --- Monthly Results ---
                if monthly_results:
                    monthly_df = pd.concat(monthly_results, ignore_index=True)
                    monthly_df = monthly_df.sort_values(["STATION", "YEAR", "MONTH"])
                    monthly_df["GPH_OPT"] = monthly_df["GPH_OPT"].astype(int)
                    monthly_df["GPH_BASE"] = monthly_df["GPH_BASE"].astype(int)
                    monthly_df["GPH_CONS"] = monthly_df["GPH_CONS"].astype(int)

                    st.subheader("Monthly GPH Aggregates")
                    st.dataframe(monthly_df, use_container_width=True)

                    if export_format == "CSV":
                        st.download_button(
                            "Download Monthly CSV",
                            data=monthly_df.to_csv(index=False),
                            file_name=f"GPH_Monthly_{ts}.csv",
                            mime="text/csv",
                            key="dl_monthly_csv",
                        )
                    else:
                        buf2 = BytesIO()
                        monthly_df.to_excel(buf2, index=False, engine="openpyxl")
                        st.download_button(
                            "Download Monthly Excel",
                            data=buf2.getvalue(),
                            file_name=f"GPH_Monthly_{ts}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="dl_monthly_xlsx",
                        )
            else:
                st.error("No results to display")


if __name__ == "__main__":
    main()
