"""
Extreme Weather & Climate Projection Tools for Azure AI Agent Service

Standalone module that fetches NASA NEX-GDDP-CMIP6 climate data directly
from Planetary Computer STAC. No prior STAC search or session state required.

Data format: NetCDF (not COG) — point sampling only, no tile rendering.
Grid resolution: 0.25° × 0.25° global
Longitude convention: 0–360 (negative longitudes are converted automatically)

Climate variables:
  tasmax, tasmin, tas — temperature (K -> °F)
  pr — precipitation (kg/m²/s -> mm/day)
  sfcWind — wind speed (m/s)
  hurs — relative humidity (%)
  huss — specific humidity (kg/kg)
  rlds — downwelling longwave radiation (W/m²)
  rsds — downwelling shortwave radiation (W/m²)

Usage:
    from geoint.extreme_weather_tools import create_extreme_weather_functions
    functions = create_extreme_weather_functions()  # Returns Set[Callable]
    tool = FunctionTool(functions)
"""

import logging
import json
import time
from typing import Dict, Any, List, Set, Callable, Optional

from cloud_config import cloud_cfg

logger = logging.getLogger(__name__)

# Module-level STAC catalog (lazy-loaded)
_catalog = None
_stac_endpoint = cloud_cfg.stac_catalog_url

# NEX-GDDP-CMIP6 collection ID
CMIP6_COLLECTION = "nasa-nex-gddp-cmip6"

# Climate variable metadata
CLIMATE_VAR_INFO: Dict[str, Dict[str, Any]] = {
    'tas': {
        'name': 'Daily Near-Surface Air Temperature',
        'unit': 'K', 'display_unit': '°F',
        'convert': lambda v: round((v - 273.15) * 9/5 + 32, 1),
        'valid_range': (150, 350),
        'category': 'temperature',
    },
    'tasmax': {
        'name': 'Daily Maximum Temperature',
        'unit': 'K', 'display_unit': '°F',
        'convert': lambda v: round((v - 273.15) * 9/5 + 32, 1),
        'valid_range': (150, 380),
        'category': 'temperature',
    },
    'tasmin': {
        'name': 'Daily Minimum Temperature',
        'unit': 'K', 'display_unit': '°F',
        'convert': lambda v: round((v - 273.15) * 9/5 + 32, 1),
        'valid_range': (150, 350),
        'category': 'temperature',
    },
    'pr': {
        'name': 'Precipitation',
        'unit': 'kg m⁻² s⁻¹', 'display_unit': 'mm/day',
        'convert': lambda v: round(v * 86400, 2),
        'valid_range': (0, 1),
        'category': 'precipitation',
    },
    'sfcWind': {
        'name': 'Near-Surface Wind Speed',
        'unit': 'm/s', 'display_unit': 'm/s',
        'convert': lambda v: round(v, 2),
        'valid_range': (0, 100),
        'category': 'wind',
    },
    'hurs': {
        'name': 'Near-Surface Relative Humidity',
        'unit': '%', 'display_unit': '%',
        'convert': lambda v: round(v, 1),
        'valid_range': (0, 100),
        'category': 'humidity',
    },
    'huss': {
        'name': 'Near-Surface Specific Humidity',
        'unit': 'kg/kg', 'display_unit': 'g/kg',
        'convert': lambda v: round(v * 1000, 3),
        'valid_range': (0, 0.1),
        'category': 'humidity',
    },
    'rlds': {
        'name': 'Downwelling Longwave Radiation',
        'unit': 'W/m²', 'display_unit': 'W/m²',
        'convert': lambda v: round(v, 1),
        'valid_range': (0, 600),
        'category': 'radiation',
    },
    'rsds': {
        'name': 'Downwelling Shortwave Radiation',
        'unit': 'W/m²', 'display_unit': 'W/m²',
        'convert': lambda v: round(v, 1),
        'valid_range': (0, 500),
        'category': 'radiation',
    },
}

# Default models/scenarios to search
PREFERRED_MODELS = ['ACCESS-CM2', 'GFDL-ESM4', 'MPI-ESM1-2-HR', 'UKESM1-0-LL', 'EC-Earth3']
PREFERRED_SCENARIOS = ['ssp245', 'ssp585']  # Moderate & worst-case


def _get_catalog():
    """Lazy-load STAC catalog."""
    global _catalog
    if _catalog is None:
        from pystac_client import Client
        _catalog = Client.open(_stac_endpoint)
    return _catalog


def _convert_longitude(longitude: float) -> float:
    """Convert standard longitude (-180..180) to NEX-GDDP convention (0..360)."""
    return longitude if longitude >= 0 else longitude + 360


def _search_cmip6_items(
    latitude: float,
    longitude: float,
    variable: str,
    scenario: str = "ssp585",
    year: Optional[int] = None,
    limit: int = 5,
) -> list:
    """
    Search Planetary Computer for NEX-GDDP-CMIP6 items.
    
    Returns raw STAC items matching the given scenario and year.
    
    CMIP6 items are GLOBAL (bbox: [-180,-90,180,90]) and contain ALL climate
    variables as separate assets (pr, tas, tasmax, tasmin, sfcWind, hurs, huss,
    rlds, rsds). Item ID format: MODEL.scenario.year (e.g. ACCESS-CM2.ssp585.2050).
    
    Filterable properties: cmip6:year, cmip6:model, cmip6:scenario
    (NOT cmip6:variable — variables are asset keys, not item properties).
    """
    import httpx
    import planetary_computer as pc

    target_year = year if year else 2030
    
    search_body: Dict[str, Any] = {
        "collections": [CMIP6_COLLECTION],
        "limit": limit,
    }
    
    # Filter by scenario and year using actual CMIP6 properties
    # (NOT cmip6:variable — that property does not exist; variables are assets)
    search_body["query"] = {
        "cmip6:year": {"eq": target_year},
    }
    if scenario:
        search_body["query"]["cmip6:scenario"] = {"eq": scenario}
    
    # Prefer well-known models for more reliable results
    # Filter to preferred models if possible
    if PREFERRED_MODELS:
        search_body["query"]["cmip6:model"] = {"in": PREFERRED_MODELS}
    
    # NOTE: No spatial filter needed — all CMIP6 items are global grids.
    # NOTE: No datetime filter — items have null datetime; use cmip6:year instead.
    
    try:
        logger.info(f"[CMIP6] Searching for {variable} asset in {scenario}/{target_year} items (limit={limit})")
        
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{_stac_endpoint}/search",
                json=search_body,
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code == 200:
                features = resp.json().get("features", [])
                logger.info(f"[CMIP6] Found {len(features)} items for {scenario}/{target_year}")
                
                # Filter to items that actually have the requested variable as an asset
                valid_features = []
                for f in features:
                    if variable in f.get("assets", {}):
                        try:
                            valid_features.append(pc.sign(f))
                        except Exception:
                            valid_features.append(f)
                
                if not valid_features and features:
                    logger.warning(f"[CMIP6] {len(features)} items found but none have '{variable}' asset")
                    # Log available assets from first item for debugging
                    first_assets = list(features[0].get("assets", {}).keys())
                    logger.warning(f"[CMIP6] Available assets in first item: {first_assets}")
                
                logger.info(f"[CMIP6] {len(valid_features)} items have '{variable}' asset")
                return valid_features
            else:
                logger.warning(f"STAC search returned {resp.status_code}: {resp.text[:200]}")
                return []
    except Exception as e:
        logger.error(f"CMIP6 STAC search failed: {e}")
        return []


def _sample_netcdf(
    href: str,
    variable: str,
    latitude: float,
    longitude: float,
    aggregate: str = "last",
) -> Dict[str, Any]:
    """
    Sample a single NetCDF asset at (lat, lon) using xarray + h5netcdf.
    
    Uses xarray with h5netcdf engine and fsspec for remote HTTP access.
    This avoids GDAL's netCDF driver which requires userfaultfd (blocked
    by Docker's default seccomp profile in Azure Container Apps).
    
    Args:
        aggregate: How to aggregate across time dimension.
            "last"  — single value from the last timestep (good for temperature)
            "annual" — mean, max, min across all timesteps (good for precip/wind)
    
    Returns dict with 'raw_value', 'display_value', 'display_unit', etc.
    or 'error' key on failure.
    """
    import xarray as xr
    import fsspec
    import numpy as np

    var_info = CLIMATE_VAR_INFO.get(variable, {
        'name': variable, 'unit': 'raw', 'display_unit': '',
        'convert': lambda v: round(v, 4), 'valid_range': None,
    })

    sample_lng = _convert_longitude(longitude)

    logger.info(f"[CMIP6] Sampling NetCDF: variable={variable}, lat={latitude}, lng={longitude}, sample_lng={sample_lng}, aggregate={aggregate}")
    logger.info(f"[CMIP6] href (first 120 chars): {href[:120]}...")

    try:
        # Open remote NetCDF via fsspec HTTP filesystem + h5netcdf engine
        # This bypasses GDAL entirely — no userfaultfd needed
        # decode_times=False avoids cftime dependency for non-standard calendars
        # (e.g. UKESM1-0-LL uses 360_day calendar)
        #
        # Use HTTPFileSystem for byte-range reads — avoids downloading the
        # entire 100-250 MB NetCDF file when we only need a single grid cell.
        fs = fsspec.filesystem("https")
        f = fs.open(href, mode="rb")
        try:
            ds = xr.open_dataset(f, engine="h5netcdf", decode_times=False)

            if variable not in ds.data_vars:
                available = list(ds.data_vars.keys())
                return {"error": f"Variable '{variable}' not found. Available: {available}"}

            var_data = ds[variable]

            # Select nearest grid cell to the target coordinates
            # NEX-GDDP-CMIP6 uses 'lat' and 'lon' dimension names
            try:
                point = var_data.sel(lat=latitude, lon=sample_lng, method="nearest")
            except KeyError:
                # Try alternate dimension names
                dim_names = list(var_data.dims)
                logger.warning(f"[CMIP6] Unexpected dims: {dim_names}, trying positional selection")
                return {"error": f"Cannot map coordinates to dimensions: {dim_names}"}

            has_time = "time" in point.dims
            total_timesteps = point.sizes["time"] if has_time else 1

            if aggregate == "annual" and has_time:
                # Compute annual statistics across all days in the file
                # Drop NaN values before computing stats
                all_values = point.values.astype(float)
                valid_mask = ~np.isnan(all_values)
                if not valid_mask.any():
                    return {"error": "No data at this location (all days masked)"}
                valid = all_values[valid_mask]

                raw_mean = float(np.mean(valid))
                raw_max = float(np.max(valid))
                raw_min = float(np.min(valid))

                display_mean = var_info['convert'](raw_mean)
                display_max = var_info['convert'](raw_max)
                display_min = var_info['convert'](raw_min)

                result = {
                    "raw_mean": round(raw_mean, 6),
                    "raw_max": round(raw_max, 6),
                    "raw_min": round(raw_min, 6),
                    "display_mean": display_mean,
                    "display_max": display_max,
                    "display_min": display_min,
                    "display_value": display_mean,
                    "display_unit": var_info['display_unit'],
                    "variable_name": var_info['name'],
                    "aggregation": "annual",
                    "days_sampled": int(valid_mask.sum()),
                    "total_days": total_timesteps,
                    "grid_resolution": "0.25° × 0.25°",
                }
                logger.info(f"[CMIP6]  NetCDF annual stats: {variable} mean={display_mean}, max={display_max}, min={display_min} {var_info['display_unit']} ({int(valid_mask.sum())} days)")
                return result

            else:
                # Single timestep: last day
                if has_time:
                    point = point.isel(time=-1)

                raw_value = float(point.values)

                # Check for NaN (masked/fill values become NaN in xarray)
                if np.isnan(raw_value):
                    return {"error": "No data at this location (masked)"}

                # Validate raw value against expected range
                vr = var_info.get('valid_range')
                if vr and not (vr[0] <= raw_value <= vr[1]):
                    return {"error": f"Value {raw_value} outside valid range {vr}"}

                display_value = var_info['convert'](raw_value)

                result = {
                    "raw_value": round(raw_value, 4),
                    "display_value": display_value,
                    "display_unit": var_info['display_unit'],
                    "variable_name": var_info['name'],
                    "band_sampled": total_timesteps,
                    "total_bands": total_timesteps,
                    "grid_resolution": "0.25° × 0.25°",
                }
                logger.info(f"[CMIP6]  NetCDF sampled OK: {variable}={display_value}{var_info['display_unit']} (raw={raw_value:.4f}, timestep={total_timesteps})")
                return result
        finally:
            try:
                f.close()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"[CMIP6]  NetCDF sampling FAILED for {variable}: {type(e).__name__}: {e}")
        logger.error(f"[CMIP6]   href={href[:250]}")
        import traceback
        logger.error(f"[CMIP6]   traceback: {traceback.format_exc()[-500:]}")
        return {"error": str(e)}


# ============================================================
# PUBLIC TOOL FUNCTIONS (registered with Agent Service)
# ============================================================

def get_temperature_projection(latitude: float, longitude: float, scenario: str = "ssp585", year: int = 2030) -> str:
    """Get projected temperature data (max, min, mean) for a location from NASA NEX-GDDP-CMIP6 climate models.
    Returns daily maximum temperature, minimum temperature, and mean temperature in °F.
    Use this when the user asks about future temperatures, heat waves, warming, or thermal conditions.
    
    :param latitude: Latitude of the location to analyze
    :param longitude: Longitude of the location to analyze  
    :param scenario: SSP scenario - 'ssp245' (moderate) or 'ssp585' (worst-case). Default 'ssp585'
    :param year: Projection year (2015-2100). Default 2030
    :return: JSON string with projected temperature values and model metadata
    """
    try:
        logger.info(f"[TOOL] get_temperature_projection at ({latitude:.4f}, {longitude:.4f}), {scenario}, {year}")
        
        temp_vars = ['tasmax', 'tasmin', 'tas']
        results = {}
        models_used = set()
        
        for var in temp_vars:
            items = _search_cmip6_items(latitude, longitude, var, scenario, year, limit=3)
            if not items:
                results[var] = {"error": f"No CMIP6 data found for {var}"}
                continue
            
            # Sample the first available item
            sampled = False
            for item in items:
                assets = item.get('assets', {})
                href = assets.get(var, {}).get('href', '') if isinstance(assets.get(var), dict) else ''
                if not href:
                    continue
                
                sample = _sample_netcdf(href, var, latitude, longitude)
                if 'error' not in sample:
                    var_info = CLIMATE_VAR_INFO[var]
                    results[var] = {
                        "value": sample['display_value'],
                        "unit": sample['display_unit'],
                        "description": var_info['name'],
                    }
                    # Extract model name from item ID  
                    item_id = item.get('id', '')
                    parts = item_id.split('.')
                    if parts:
                        models_used.add(parts[0])
                    sampled = True
                    break
            
            if not sampled:
                results[var] = {"error": f"Sampling failed for {var}"}
        
        output = {
            "location": {"latitude": latitude, "longitude": longitude},
            "scenario": scenario,
            "year": year,
            "data_source": "NASA NEX-GDDP-CMIP6",
            "grid_resolution": "0.25° × 0.25°",
            "models_sampled": list(models_used),
            "projections": {}
        }
        
        if 'tasmax' in results and 'error' not in results.get('tasmax', {}):
            output["projections"]["daily_max_temperature"] = results['tasmax']
        if 'tasmin' in results and 'error' not in results.get('tasmin', {}):
            output["projections"]["daily_min_temperature"] = results['tasmin']
        if 'tas' in results and 'error' not in results.get('tas', {}):
            output["projections"]["daily_mean_temperature"] = results['tas']
        
        if not output["projections"]:
            output["error"] = "Could not retrieve temperature data. " + json.dumps(results)
        
        logger.info(f"[TOOL] Temperature projection: {json.dumps(output.get('projections', {}))}")
        return json.dumps(output)
        
    except Exception as e:
        logger.error(f"[TOOL] Temperature projection failed: {e}")
        return json.dumps({"error": str(e)})


def get_precipitation_projection(latitude: float, longitude: float, scenario: str = "ssp585", year: int = 2030) -> str:
    """Get projected precipitation (rainfall) data for a location from NASA NEX-GDDP-CMIP6 climate models.
    Returns daily precipitation in mm/day.
    Use this when the user asks about future rainfall, drought, flooding risk, or precipitation patterns.
    
    :param latitude: Latitude of the location to analyze
    :param longitude: Longitude of the location to analyze
    :param scenario: SSP scenario - 'ssp245' (moderate) or 'ssp585' (worst-case). Default 'ssp585'
    :param year: Projection year (2015-2100). Default 2030
    :return: JSON string with projected precipitation values and model metadata
    """
    try:
        logger.info(f"[TOOL] get_precipitation_projection at ({latitude:.4f}, {longitude:.4f}), {scenario}, {year}")
        
        items = _search_cmip6_items(latitude, longitude, 'pr', scenario, year, limit=5)
        
        if not items:
            return json.dumps({
                "error": "No CMIP6 precipitation data found for this location/scenario",
                "location": {"latitude": latitude, "longitude": longitude},
                "scenario": scenario, "year": year,
            })
        
        # Sample multiple models for ensemble view
        model_results = []
        for item in items[:3]:
            assets = item.get('assets', {})
            href = assets.get('pr', {}).get('href', '') if isinstance(assets.get('pr'), dict) else ''
            if not href:
                continue
            
            sample = _sample_netcdf(href, 'pr', latitude, longitude, aggregate="annual")
            if 'error' not in sample:
                item_id = item.get('id', '')
                parts = item_id.split('.')
                model_name = parts[0] if parts else 'Unknown'
                model_results.append({
                    "model": model_name,
                    "mean_precipitation_mm_per_day": sample.get('display_mean', sample.get('display_value')),
                    "max_precipitation_mm_per_day": sample.get('display_max'),
                    "min_precipitation_mm_per_day": sample.get('display_min'),
                    "unit": sample['display_unit'],
                })
        
        if not model_results:
            return json.dumps({
                "error": "Sampling failed for all available precipitation items",
                "location": {"latitude": latitude, "longitude": longitude},
            })
        
        # Compute ensemble summary
        mean_values = [r['mean_precipitation_mm_per_day'] for r in model_results]
        max_values = [r['max_precipitation_mm_per_day'] for r in model_results if r.get('max_precipitation_mm_per_day') is not None]
        
        output = {
            "location": {"latitude": latitude, "longitude": longitude},
            "scenario": scenario,
            "year": year,
            "data_source": "NASA NEX-GDDP-CMIP6",
            "grid_resolution": "0.25° × 0.25°",
            "note": "Annual statistics computed across all days in the projected year",
            "precipitation": {
                "annual_mean_mm_per_day": round(sum(mean_values) / len(mean_values), 2),
                "annual_total_mm_estimate": round(sum(mean_values) / len(mean_values) * 365, 1),
                "peak_daily_mm": round(max(max_values), 2) if max_values else None,
                "models_sampled": len(model_results),
                "model_details": model_results,
            }
        }
        
        logger.info(f"[TOOL] Precipitation projection: mean={output['precipitation']['annual_mean_mm_per_day']} mm/day, peak={output['precipitation'].get('peak_daily_mm')} mm/day")
        return json.dumps(output)
        
    except Exception as e:
        logger.error(f"[TOOL] Precipitation projection failed: {e}")
        return json.dumps({"error": str(e)})


def get_wind_projection(latitude: float, longitude: float, scenario: str = "ssp585", year: int = 2030) -> str:
    """Get projected near-surface wind speed for a location from NASA NEX-GDDP-CMIP6 climate models.
    Returns wind speed in m/s.
    Use this when the user asks about future wind conditions, storms, wind energy, or extreme wind events.
    
    :param latitude: Latitude of the location to analyze
    :param longitude: Longitude of the location to analyze
    :param scenario: SSP scenario - 'ssp245' (moderate) or 'ssp585' (worst-case). Default 'ssp585'
    :param year: Projection year (2015-2100). Default 2030
    :return: JSON string with projected wind speed values and model metadata
    """
    try:
        logger.info(f"[TOOL] get_wind_projection at ({latitude:.4f}, {longitude:.4f}), {scenario}, {year}")
        
        items = _search_cmip6_items(latitude, longitude, 'sfcWind', scenario, year, limit=5)
        
        if not items:
            return json.dumps({
                "error": "No CMIP6 wind data found for this location/scenario",
                "location": {"latitude": latitude, "longitude": longitude},
                "scenario": scenario, "year": year,
            })
        
        model_results = []
        for item in items[:3]:
            assets = item.get('assets', {})
            href = assets.get('sfcWind', {}).get('href', '') if isinstance(assets.get('sfcWind'), dict) else ''
            if not href:
                continue
            
            sample = _sample_netcdf(href, 'sfcWind', latitude, longitude, aggregate="annual")
            if 'error' not in sample:
                item_id = item.get('id', '')
                parts = item_id.split('.')
                model_name = parts[0] if parts else 'Unknown'
                model_results.append({
                    "model": model_name,
                    "mean_wind_speed_m_s": sample.get('display_mean', sample.get('display_value')),
                    "max_wind_speed_m_s": sample.get('display_max'),
                    "min_wind_speed_m_s": sample.get('display_min'),
                    "unit": sample['display_unit'],
                })
        
        if not model_results:
            return json.dumps({
                "error": "Sampling failed for all available wind items",
                "location": {"latitude": latitude, "longitude": longitude},
            })
        
        mean_values = [r['mean_wind_speed_m_s'] for r in model_results]
        max_values = [r['max_wind_speed_m_s'] for r in model_results if r.get('max_wind_speed_m_s') is not None]
        
        # Classify wind severity based on annual mean
        mean_wind = sum(mean_values) / len(mean_values)
        if mean_wind < 3:
            wind_class = "Calm"
        elif mean_wind < 6:
            wind_class = "Light breeze"
        elif mean_wind < 10:
            wind_class = "Moderate wind"
        elif mean_wind < 17:
            wind_class = "Strong wind"
        else:
            wind_class = "Severe / storm-force"
        
        output = {
            "location": {"latitude": latitude, "longitude": longitude},
            "scenario": scenario,
            "year": year,
            "data_source": "NASA NEX-GDDP-CMIP6",
            "grid_resolution": "0.25° × 0.25°",
            "note": "Annual statistics computed across all days in the projected year",
            "wind": {
                "annual_mean_m_s": round(mean_wind, 2),
                "peak_daily_m_s": round(max(max_values), 2) if max_values else None,
                "classification": wind_class,
                "models_sampled": len(model_results),
                "model_details": model_results,
            }
        }
        
        logger.info(f"[TOOL] Wind projection: {mean_wind:.1f} m/s ({wind_class})")
        return json.dumps(output)
        
    except Exception as e:
        logger.error(f"[TOOL] Wind projection failed: {e}")
        return json.dumps({"error": str(e)})


def get_humidity_projection(latitude: float, longitude: float, scenario: str = "ssp585", year: int = 2030) -> str:
    """Get projected humidity data for a location from NASA NEX-GDDP-CMIP6 climate models.
    Returns near-surface relative humidity (%) and specific humidity (g/kg).
    Use this when the user asks about future humidity, heat index, moisture, or atmospheric conditions.
    
    :param latitude: Latitude of the location to analyze
    :param longitude: Longitude of the location to analyze
    :param scenario: SSP scenario - 'ssp245' (moderate) or 'ssp585' (worst-case). Default 'ssp585'
    :param year: Projection year (2015-2100). Default 2030
    :return: JSON string with projected humidity values and model metadata
    """
    try:
        logger.info(f"[TOOL] get_humidity_projection at ({latitude:.4f}, {longitude:.4f}), {scenario}, {year}")
        
        results = {}
        models_used = set()
        
        for var in ['hurs', 'huss']:
            items = _search_cmip6_items(latitude, longitude, var, scenario, year, limit=3)
            if not items:
                results[var] = {"error": f"No data found for {var}"}
                continue
            
            for item in items:
                assets = item.get('assets', {})
                href = assets.get(var, {}).get('href', '') if isinstance(assets.get(var), dict) else ''
                if not href:
                    continue
                
                sample = _sample_netcdf(href, var, latitude, longitude)
                if 'error' not in sample:
                    var_info = CLIMATE_VAR_INFO[var]
                    results[var] = {
                        "value": sample['display_value'],
                        "unit": sample['display_unit'],
                        "description": var_info['name'],
                    }
                    item_id = item.get('id', '')
                    parts = item_id.split('.')
                    if parts:
                        models_used.add(parts[0])
                    break
        
        output = {
            "location": {"latitude": latitude, "longitude": longitude},
            "scenario": scenario,
            "year": year,
            "data_source": "NASA NEX-GDDP-CMIP6",
            "grid_resolution": "0.25° × 0.25°",
            "models_sampled": list(models_used),
            "humidity": {}
        }
        
        if 'hurs' in results and 'error' not in results.get('hurs', {}):
            output["humidity"]["relative_humidity"] = results['hurs']
        if 'huss' in results and 'error' not in results.get('huss', {}):
            output["humidity"]["specific_humidity"] = results['huss']
        
        if not output["humidity"]:
            output["error"] = "Could not retrieve humidity data"
        
        logger.info(f"[TOOL] Humidity projection: {json.dumps(output.get('humidity', {}))}")
        return json.dumps(output)
        
    except Exception as e:
        logger.error(f"[TOOL] Humidity projection failed: {e}")
        return json.dumps({"error": str(e)})


def get_climate_overview(latitude: float, longitude: float, scenario: str = "ssp585", year: int = 2030) -> str:
    """Get a comprehensive climate overview for a location by sampling multiple variables at once.
    Returns temperature (max, min, mean), precipitation, wind speed, and humidity projections.
    Use this when the user asks for a general climate outlook, overall climate conditions, or 
    wants to understand the full climate picture for a location.
    
    :param latitude: Latitude of the location to analyze
    :param longitude: Longitude of the location to analyze
    :param scenario: SSP scenario - 'ssp245' (moderate) or 'ssp585' (worst-case). Default 'ssp585'
    :param year: Projection year (2015-2100). Default 2030
    :return: JSON string with multi-variable climate overview
    """
    try:
        logger.info(f"[TOOL] get_climate_overview at ({latitude:.4f}, {longitude:.4f}), {scenario}, {year}")
        
        overview_vars = ['tasmax', 'tasmin', 'tas', 'pr', 'sfcWind', 'hurs']
        overview = {}
        models_used = set()
        errors = []
        
        for var in overview_vars:
            items = _search_cmip6_items(latitude, longitude, var, scenario, year, limit=1)
            if not items:
                errors.append(f"No data for {var}")
                continue
            
            for item in items:
                assets = item.get('assets', {})
                href = assets.get(var, {}).get('href', '') if isinstance(assets.get(var), dict) else ''
                if not href:
                    continue
                
                # Use annual aggregation for rate-based variables (precip, wind)
                agg = "annual" if var in ('pr', 'sfcWind') else "last"
                sample = _sample_netcdf(href, var, latitude, longitude, aggregate=agg)
                if 'error' not in sample:
                    var_info = CLIMATE_VAR_INFO[var]
                    overview[var] = {
                        "value": sample.get('display_mean', sample.get('display_value')),
                        "unit": sample['display_unit'],
                        "description": var_info['name'],
                    }
                    if agg == "annual" and 'display_max' in sample:
                        overview[var]["peak"] = sample['display_max']
                    item_id = item.get('id', '')
                    parts = item_id.split('.')
                    if parts:
                        models_used.add(parts[0])
                    break
                else:
                    errors.append(f"{var}: {sample['error']}")
        
        # Build readable summary
        summary_parts = []
        if 'tasmax' in overview:
            summary_parts.append(f"Max Temp: {overview['tasmax']['value']}°F")
        if 'tasmin' in overview:
            summary_parts.append(f"Min Temp: {overview['tasmin']['value']}°F")
        if 'tas' in overview:
            summary_parts.append(f"Mean Temp: {overview['tas']['value']}°F")
        if 'pr' in overview:
            summary_parts.append(f"Precip: {overview['pr']['value']} mm/day")
        if 'sfcWind' in overview:
            summary_parts.append(f"Wind: {overview['sfcWind']['value']} m/s")
        if 'hurs' in overview:
            summary_parts.append(f"Humidity: {overview['hurs']['value']}%")
        
        output = {
            "location": {"latitude": latitude, "longitude": longitude},
            "scenario": scenario,
            "scenario_description": "SSP2-4.5 (moderate)" if scenario == "ssp245" else "SSP5-8.5 (worst-case)" if scenario == "ssp585" else scenario,
            "year": year,
            "data_source": "NASA NEX-GDDP-CMIP6",
            "grid_resolution": "0.25° × 0.25°",
            "models_sampled": list(models_used),
            "climate_summary": " | ".join(summary_parts) if summary_parts else "No data sampled",
            "variables": overview,
            "note": "These are climate PROJECTIONS from CMIP6 models, not observations."
        }
        
        if errors:
            output["warnings"] = errors
        
        logger.info(f"[TOOL] Climate overview: {len(overview)} variables sampled for {scenario}/{year}")
        return json.dumps(output)
        
    except Exception as e:
        logger.error(f"[TOOL] Climate overview failed: {e}")
        return json.dumps({"error": str(e)})


def compare_climate_scenarios(latitude: float, longitude: float, year: int = 2030) -> str:
    """Compare climate projections between SSP2-4.5 (moderate emissions) and SSP5-8.5 (worst-case emissions)
    scenarios for a location. Shows temperature and precipitation differences between scenarios.
    Use this when the user asks about comparing emission scenarios, best vs worst case, or climate uncertainty.
    
    :param latitude: Latitude of the location to analyze
    :param longitude: Longitude of the location to analyze
    :param year: Projection year (2015-2100). Default 2030
    :return: JSON string comparing key climate variables across both SSP scenarios
    """
    try:
        logger.info(f"[TOOL] compare_climate_scenarios at ({latitude:.4f}, {longitude:.4f}), {year}")
        
        compare_vars = ['tasmax', 'pr']
        scenarios = ['ssp245', 'ssp585']
        comparison = {}
        
        for var in compare_vars:
            comparison[var] = {}
            var_info = CLIMATE_VAR_INFO[var]
            
            for sc in scenarios:
                items = _search_cmip6_items(latitude, longitude, var, sc, year, limit=1)
                if not items:
                    comparison[var][sc] = {"error": "No data"}
                    continue
                
                for item in items:
                    assets = item.get('assets', {})
                    href = assets.get(var, {}).get('href', '') if isinstance(assets.get(var), dict) else ''
                    if not href:
                        continue
                    
                    # Use annual aggregation for rate-based variables
                    agg = "annual" if var in ('pr', 'sfcWind') else "last"
                    sample = _sample_netcdf(href, var, latitude, longitude, aggregate=agg)
                    if 'error' not in sample:
                        comparison[var][sc] = {
                            "value": sample.get('display_mean', sample.get('display_value')),
                            "unit": sample['display_unit'],
                        }
                        break
                else:
                    comparison[var][sc] = {"error": "Sampling failed"}
        
        # Calculate deltas
        deltas = {}
        for var in compare_vars:
            ssp245_val = comparison.get(var, {}).get('ssp245', {}).get('value')
            ssp585_val = comparison.get(var, {}).get('ssp585', {}).get('value')
            if ssp245_val is not None and ssp585_val is not None:
                deltas[var] = {
                    "difference": round(ssp585_val - ssp245_val, 2),
                    "unit": CLIMATE_VAR_INFO[var]['display_unit'],
                    "description": f"SSP5-8.5 minus SSP2-4.5",
                }
        
        output = {
            "location": {"latitude": latitude, "longitude": longitude},
            "year": year,
            "data_source": "NASA NEX-GDDP-CMIP6",
            "scenarios": {
                "ssp245": "SSP2-4.5 — Moderate emissions (sustainable development path)",
                "ssp585": "SSP5-8.5 — Worst-case emissions (fossil fuel intensive)",
            },
            "comparison": comparison,
            "scenario_difference": deltas,
            "note": "Positive difference = worse conditions under high emissions"
        }
        
        logger.info(f"[TOOL] Scenario comparison complete for {year}")
        return json.dumps(output)
        
    except Exception as e:
        logger.error(f"[TOOL] Scenario comparison failed: {e}")
        return json.dumps({"error": str(e)})


def get_radiation_projection(latitude: float, longitude: float, scenario: str = "ssp585", year: int = 2030) -> str:
    """Get projected solar and longwave radiation data for a location from NASA NEX-GDDP-CMIP6 models.
    Returns downwelling shortwave (solar) and longwave radiation in W/m².
    Use this when the user asks about solar energy potential, radiation budget, or energy balance.
    
    :param latitude: Latitude of the location to analyze
    :param longitude: Longitude of the location to analyze
    :param scenario: SSP scenario - 'ssp245' (moderate) or 'ssp585' (worst-case). Default 'ssp585'
    :param year: Projection year (2015-2100). Default 2030
    :return: JSON string with projected radiation values and model metadata
    """
    try:
        logger.info(f"[TOOL] get_radiation_projection at ({latitude:.4f}, {longitude:.4f}), {scenario}, {year}")
        
        results = {}
        models_used = set()
        
        for var in ['rsds', 'rlds']:
            items = _search_cmip6_items(latitude, longitude, var, scenario, year, limit=3)
            if not items:
                results[var] = {"error": f"No data found for {var}"}
                continue
            
            for item in items:
                assets = item.get('assets', {})
                href = assets.get(var, {}).get('href', '') if isinstance(assets.get(var), dict) else ''
                if not href:
                    continue
                
                sample = _sample_netcdf(href, var, latitude, longitude)
                if 'error' not in sample:
                    var_info = CLIMATE_VAR_INFO[var]
                    results[var] = {
                        "value": sample['display_value'],
                        "unit": sample['display_unit'],
                        "description": var_info['name'],
                    }
                    item_id = item.get('id', '')
                    parts = item_id.split('.')
                    if parts:
                        models_used.add(parts[0])
                    break
        
        output = {
            "location": {"latitude": latitude, "longitude": longitude},
            "scenario": scenario,
            "year": year,
            "data_source": "NASA NEX-GDDP-CMIP6",
            "grid_resolution": "0.25° × 0.25°",
            "models_sampled": list(models_used),
            "radiation": {}
        }
        
        if 'rsds' in results and 'error' not in results.get('rsds', {}):
            output["radiation"]["shortwave_solar"] = results['rsds']
        if 'rlds' in results and 'error' not in results.get('rlds', {}):
            output["radiation"]["longwave"] = results['rlds']
        
        if not output["radiation"]:
            output["error"] = "Could not retrieve radiation data"
        
        logger.info(f"[TOOL] Radiation projection: {json.dumps(output.get('radiation', {}))}")
        return json.dumps(output)
        
    except Exception as e:
        logger.error(f"[TOOL] Radiation projection failed: {e}")
        return json.dumps({"error": str(e)})


def create_extreme_weather_functions() -> Set[Callable]:
    """Create the set of extreme weather/climate analysis functions for FunctionTool.
    
    Returns a Set[Callable] that can be passed to FunctionTool().
    Each function uses docstring-based parameter descriptions.
    """
    return {
        get_temperature_projection,
        get_precipitation_projection,
        get_wind_projection,
        get_humidity_projection,
        get_climate_overview,
        compare_climate_scenarios,
        get_radiation_projection,
    }
