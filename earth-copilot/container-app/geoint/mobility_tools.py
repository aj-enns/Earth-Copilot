"""
GEOINT Mobility Analysis Tools for Azure AI Agent Service

Standalone synchronous functions compatible with Azure AI Agent Service FunctionTool.
Each function uses docstring-based parameter descriptions and returns JSON strings.

IMPORTANT: All functions are fully synchronous (no asyncio wrappers) to avoid
event-loop conflicts when the Agent SDK calls them from its own async context.

Usage:
    from geoint.mobility_tools import create_mobility_functions
    functions = create_mobility_functions()  # Returns Set[Callable]
    tool = AsyncFunctionTool(functions)
"""

import logging
import json
import math
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta

import numpy as np
import planetary_computer
import pystac_client
from cloud_config import cloud_cfg

logger = logging.getLogger(__name__)

# Module-level constants
STAC_ENDPOINT = cloud_cfg.stac_catalog_url
RADIUS_MILES = 5
SLOPE_THRESHOLD_SLOW = 15   # degrees
SLOPE_THRESHOLD_NO_GO = 30  # degrees
WATER_BACKSCATTER_THRESHOLD = -20  # dB
VEGETATION_NDVI_DENSE = 0.6
FIRE_CONFIDENCE_THRESHOLD = 50

# Lazy-loaded STAC catalog
_catalog = None


def _get_catalog():
    """Lazy-load STAC catalog (synchronous pystac_client)."""
    global _catalog
    if _catalog is None:
        _catalog = pystac_client.Client.open(STAC_ENDPOINT)
    return _catalog


def _convert_numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_python(item) for item in obj)
    return obj


def _calculate_bbox(latitude: float, longitude: float, radius_miles: float = 5.0) -> List[float]:
    """Calculate bounding box from center point and radius in miles."""
    lat_delta = radius_miles / 69.0
    lon_delta = radius_miles / (69.0 * math.cos(math.radians(latitude)))
    return [
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta
    ]


def _calculate_directional_bbox(latitude: float, longitude: float, cardinal: str) -> List[float]:
    """Calculate bounding box for a cardinal direction sector (N/S/E/W)."""
    sector_radius = RADIUS_MILES / 2.0
    lat_delta = sector_radius / 69.0
    lon_delta = sector_radius / (69.0 * math.cos(math.radians(latitude)))

    if cardinal == "N":
        return [longitude - lon_delta, latitude, longitude + lon_delta, latitude + lat_delta]
    elif cardinal == "S":
        return [longitude - lon_delta, latitude - lat_delta, longitude + lon_delta, latitude]
    elif cardinal == "E":
        return [longitude, latitude - lat_delta, longitude + lon_delta, latitude + lat_delta]
    else:  # W
        return [longitude - lon_delta, latitude - lat_delta, longitude, latitude + lat_delta]


def _query_stac_collection_sync(
    collection: str, bbox: List[float],
    datetime_range: Optional[str] = None,
    query_params: Optional[Dict] = None,
    limit: int = 10
) -> list:
    """Query a STAC collection synchronously via pystac_client."""
    try:
        catalog = _get_catalog()
        search_params = {
            "collections": [collection],
            "bbox": bbox,
            "limit": limit,
        }
        if datetime_range:
            search_params["datetime"] = datetime_range
        if query_params:
            search_params["query"] = query_params

        search = catalog.search(**search_params)
        items = list(search.items())
        return [planetary_computer.sign(item) for item in items]
    except Exception as e:
        logger.error(f"STAC query error for {collection}: {e}")
        return []


def _read_cog_window_sync(asset_url: str, bbox: List[float], band: int = 1) -> Optional[np.ndarray]:
    """Read pixels from a Cloud-Optimized GeoTIFF for a bounding box (synchronous)."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.warp import transform_bounds

        signed_url = planetary_computer.sign_url(asset_url)
        with rasterio.open(signed_url) as src:
            # Reproject bbox from EPSG:4326 to raster's native CRS if needed
            if src.crs and str(src.crs) != "EPSG:4326":
                reprojected = transform_bounds("EPSG:4326", src.crs, *bbox)
            else:
                reprojected = bbox
            window = from_bounds(*reprojected, src.transform)
            data = src.read(band, window=window)
            if src.nodata is not None:
                data = data.astype(float)
                data[data == src.nodata] = np.nan
            return data
    except Exception as e:
        logger.error(f"Failed to read COG: {e}")
        return None


def _analyze_fire_pixels(pixels: np.ndarray) -> Dict[str, Any]:
    """Analyze MODIS FireMask pixel values."""
    valid = pixels[~np.isnan(pixels)]
    if len(valid) == 0:
        return {"status": "GO", "reason": "No fire data available", "confidence": "low"}
    high = int(np.sum(valid == 9))
    nominal = int(np.sum(valid == 8))
    low = int(np.sum(valid == 7))
    total = high + nominal + low
    if high > 0:
        return {"status": "NO-GO", "reason": f"Active fires detected: {high} high-confidence fire pixels", "confidence": "high", "metrics": {"high": high, "nominal": nominal, "low": low, "total": total}}
    elif total > 5:
        return {"status": "SLOW-GO", "reason": f"Multiple fire detections: {total} pixels", "confidence": "medium", "metrics": {"high": high, "nominal": nominal, "low": low, "total": total}}
    elif total > 0:
        return {"status": "SLOW-GO", "reason": f"Potential fires: {total} low-confidence detections", "confidence": "medium", "metrics": {"high": high, "nominal": nominal, "low": low, "total": total}}
    return {"status": "GO", "reason": "No active fires detected", "confidence": "high", "metrics": {"total": 0}}


def _analyze_water_pixels(pixels: np.ndarray) -> Dict[str, Any]:
    """Analyze SAR VV backscatter for water bodies."""
    valid = pixels[~np.isnan(pixels)]
    if len(valid) == 0:
        return {"status": "GO", "reason": "No SAR data available", "confidence": "low"}
    water_pct = float((np.sum(valid < WATER_BACKSCATTER_THRESHOLD) / len(valid)) * 100)
    if water_pct > 30:
        return {"status": "NO-GO", "reason": f"Major water bodies: {water_pct:.1f}% coverage", "confidence": "high", "metrics": {"water_pct": round(water_pct, 1)}}
    elif water_pct > 10:
        return {"status": "SLOW-GO", "reason": f"Moderate water coverage: {water_pct:.1f}%", "confidence": "high", "metrics": {"water_pct": round(water_pct, 1)}}
    return {"status": "GO", "reason": f"Minimal water: {water_pct:.1f}% coverage", "confidence": "high", "metrics": {"water_pct": round(water_pct, 1)}}


def _analyze_jrc_water_pixels(pixels: np.ndarray) -> Dict[str, Any]:
    """Analyze JRC Global Surface Water occurrence data.
    Pixel values: 0-100 = % of time water was observed (1984-2021).
    Values 0 or nodata = never water. 100 = permanent water.
    """
    valid = pixels[~np.isnan(pixels)]
    if len(valid) == 0:
        return {"status": "GO", "reason": "No water occurrence data available", "confidence": "low"}
    # Pixels with occurrence >= 50% are considered water bodies
    water_pixels = np.sum(valid >= 50)
    water_pct = float(water_pixels / len(valid) * 100)
    avg_occurrence = float(np.mean(valid[valid > 0])) if np.any(valid > 0) else 0.0
    permanent_pct = float(np.sum(valid >= 80) / len(valid) * 100)  # Near-permanent water
    if water_pct > 30:
        return {"status": "NO-GO", "reason": f"Major water bodies: {water_pct:.1f}% coverage ({permanent_pct:.1f}% permanent)", "confidence": "high", "metrics": {"water_pct": round(water_pct, 1), "permanent_pct": round(permanent_pct, 1), "avg_occurrence": round(avg_occurrence, 1)}}
    elif water_pct > 10:
        return {"status": "SLOW-GO", "reason": f"Moderate water coverage: {water_pct:.1f}% (avg occurrence {avg_occurrence:.0f}%)", "confidence": "high", "metrics": {"water_pct": round(water_pct, 1), "permanent_pct": round(permanent_pct, 1), "avg_occurrence": round(avg_occurrence, 1)}}
    return {"status": "GO", "reason": f"Minimal water: {water_pct:.1f}% coverage", "confidence": "high", "metrics": {"water_pct": round(water_pct, 1), "permanent_pct": round(permanent_pct, 1), "avg_occurrence": round(avg_occurrence, 1)}}


def _analyze_elevation_pixels(pixels: np.ndarray) -> Dict[str, Any]:
    """Analyze DEM elevation for slope classification."""
    valid = pixels[~np.isnan(pixels)]
    if len(valid) == 0:
        return {"status": "GO", "reason": "No elevation data", "confidence": "low"}
    dy, dx = np.gradient(pixels, 30)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    valid_slopes = slope_deg[~np.isnan(slope_deg)]
    if len(valid_slopes) == 0:
        return {"status": "GO", "reason": "Unable to calculate slopes", "confidence": "low"}
    total = len(valid_slopes)
    gentle = float(np.sum(valid_slopes < 15) / total * 100)
    moderate = float(np.sum((valid_slopes >= 15) & (valid_slopes < 30)) / total * 100)
    steep = float(np.sum(valid_slopes >= 30) / total * 100)
    max_s = float(np.max(valid_slopes))
    avg_s = float(np.mean(valid_slopes))
    if steep > 30:
        return {"status": "NO-GO", "reason": f"Steep terrain: {steep:.1f}% > 30 deg slopes (max {max_s:.1f} deg)", "confidence": "high", "metrics": {"avg": round(avg_s, 1), "max": round(max_s, 1), "gentle_pct": round(gentle, 1), "moderate_pct": round(moderate, 1), "steep_pct": round(steep, 1)}}
    elif moderate + steep > 50:
        return {"status": "SLOW-GO", "reason": f"Moderate terrain: {moderate:.1f}% slopes 15-30 deg (avg {avg_s:.1f} deg)", "confidence": "high", "metrics": {"avg": round(avg_s, 1), "max": round(max_s, 1), "gentle_pct": round(gentle, 1), "moderate_pct": round(moderate, 1), "steep_pct": round(steep, 1)}}
    return {"status": "GO", "reason": f"Gentle terrain: {gentle:.1f}% < 15 deg slopes (avg {avg_s:.1f} deg)", "confidence": "high", "metrics": {"avg": round(avg_s, 1), "max": round(max_s, 1), "gentle_pct": round(gentle, 1), "moderate_pct": round(moderate, 1), "steep_pct": round(steep, 1)}}


def _analyze_vegetation_pixels(red: np.ndarray, nir: np.ndarray) -> Dict[str, Any]:
    """Analyze NDVI from Sentinel-2 red/NIR bands."""
    ndvi = (nir - red) / (nir + red + 1e-8)
    valid = ndvi[(~np.isnan(ndvi)) & (ndvi >= -1) & (ndvi <= 1)]
    if len(valid) == 0:
        return {"status": "GO", "reason": "No vegetation data", "confidence": "low"}
    total = len(valid)
    sparse = float(np.sum(valid < 0.5) / total * 100)
    moderate = float(np.sum((valid >= 0.5) & (valid < 0.7)) / total * 100)
    dense = float(np.sum(valid >= 0.7) / total * 100)
    avg = float(np.mean(valid))
    if dense > 50:
        return {"status": "NO-GO", "reason": f"Dense vegetation: {dense:.1f}% (NDVI > 0.7)", "confidence": "high", "metrics": {"avg_ndvi": round(avg, 2), "dense_pct": round(dense, 1)}}
    elif moderate + dense > 60:
        return {"status": "SLOW-GO", "reason": f"Moderate vegetation: {moderate:.1f}% (avg NDVI {avg:.2f})", "confidence": "high", "metrics": {"avg_ndvi": round(avg, 2), "dense_pct": round(dense, 1)}}
    return {"status": "GO", "reason": f"Light vegetation: {sparse:.1f}% sparse (avg NDVI {avg:.2f})", "confidence": "high", "metrics": {"avg_ndvi": round(avg, 2), "dense_pct": round(dense, 1)}}


# ============================================================================
# PUBLIC TOOL FUNCTIONS (registered with AsyncFunctionTool)
# All functions are fully synchronous — no asyncio wrappers.
# ============================================================================

def analyze_directional_mobility(latitude: float, longitude: float) -> str:
    """Analyze terrain mobility in all four cardinal directions (N, S, E, W) from a location.
    Returns GO / SLOW-GO / NO-GO status for each direction based on fire, water, slope, and vegetation.
    Use this when the user asks about mobility, trafficability, or ground movement.

    :param latitude: Center latitude of the analysis area
    :param longitude: Center longitude of the analysis area
    :return: JSON string with directional mobility assessments (north, south, east, west)
    """
    try:
        logger.info(f"[TOOL] analyze_directional_mobility at ({latitude:.4f}, {longitude:.4f})")
        result = _analyze_all_directions_sync(latitude, longitude)
        return json.dumps(_convert_numpy_to_python(result))
    except Exception as e:
        logger.error(f"[TOOL] analyze_directional_mobility failed: {e}")
        return json.dumps({"error": str(e)})


def _analyze_all_directions_sync(latitude: float, longitude: float) -> Dict[str, Any]:
    """Synchronous directional mobility analysis."""
    bbox = _calculate_bbox(latitude, longitude, RADIUS_MILES)

    end_date = datetime.utcnow()
    recent = f"{(end_date - timedelta(days=90)).isoformat()}Z/{end_date.isoformat()}Z"

    # Query each collection synchronously
    collection_queries = [
        ("jrc-gsw", None, None),
        ("sentinel-1-rtc", recent, None),
        ("sentinel-2-l2a", recent, {"eo:cloud_cover": {"lt": 50}}),
        ("cop-dem-glo-30", None, None),
        ("modis-14A1-061", None, None),
    ]

    collection_names = ["jrc-gsw", "sentinel-1-rtc", "sentinel-2-l2a", "cop-dem-glo-30", "modis-14A1-061"]
    data_keys = ["water_detection", "terrain_backscatter", "vegetation_density", "elevation_profile", "active_fires"]
    terrain_data = {k: None for k in data_keys}
    terrain_data["collection_status"] = {}
    terrain_data["sources"] = []

    for (col, dt_range, qparams), key in zip(collection_queries, data_keys):
        try:
            items = _query_stac_collection_sync(col, bbox, datetime_range=dt_range, query_params=qparams, limit=10)
            if items:
                terrain_data[key] = {"items_found": len(items), "collection": col, "items": items[:3]}
                terrain_data["collection_status"][col] = "success"
                terrain_data["sources"].append(col)
            else:
                terrain_data["collection_status"][col] = "no_data"
        except Exception as e:
            terrain_data["collection_status"][col] = "error"
            logger.error(f"Collection {col} query failed: {e}")

    # Analyze each direction
    directions = {}
    for name, cardinal in [("north", "N"), ("south", "S"), ("east", "E"), ("west", "W")]:
        directions[name] = _analyze_single_direction_sync(name.title(), latitude, longitude, terrain_data, cardinal)

    return {
        "location": {"latitude": latitude, "longitude": longitude},
        "radius_miles": RADIUS_MILES,
        "directions": directions,
        "data_sources": terrain_data["sources"],
        "collection_status": terrain_data["collection_status"]
    }


def _analyze_single_direction_sync(direction_name: str, lat: float, lon: float, terrain_data: Dict, cardinal: str) -> Dict:
    """Analyze mobility for one cardinal direction (synchronous)."""
    status = "GO"
    factors = []
    confidence = "medium"
    data_used = []
    metrics = {}
    d_bbox = _calculate_directional_bbox(lat, lon, cardinal)

    # Fire
    if terrain_data.get("active_fires") and terrain_data["active_fires"].get("items"):
        try:
            item = terrain_data["active_fires"]["items"][0]
            url = item.assets.get("FireMask", None)
            if url:
                px = _read_cog_window_sync(url.href, d_bbox)
                if px is not None:
                    r = _analyze_fire_pixels(px)
                    status = r["status"]
                    factors.append(r["reason"])
                    confidence = r["confidence"]
                    data_used.append("MODIS Fire")
                    metrics["fire"] = r.get("metrics", {})
        except Exception as e:
            logger.error(f"Fire analysis failed: {e}")

    # Water (JRC Global Surface Water occurrence)
    if status != "NO-GO" and terrain_data.get("water_detection") and terrain_data["water_detection"].get("items"):
        try:
            item = terrain_data["water_detection"]["items"][0]
            asset = item.assets.get("occurrence", None)
            if asset:
                px = _read_cog_window_sync(asset.href, d_bbox)
                if px is not None:
                    r = _analyze_jrc_water_pixels(px)
                    if r["status"] == "NO-GO":
                        status = "NO-GO"
                    elif r["status"] == "SLOW-GO" and status == "GO":
                        status = "SLOW-GO"
                    factors.append(r["reason"])
                    data_used.append("JRC Global Surface Water")
                    metrics["water"] = r.get("metrics", {})
        except Exception as e:
            logger.error(f"Water analysis failed: {e}")

    # Elevation/Slope
    if status == "GO" and terrain_data.get("elevation_profile") and terrain_data["elevation_profile"].get("items"):
        try:
            item = terrain_data["elevation_profile"]["items"][0]
            asset = item.assets.get("data", None)
            if asset:
                px = _read_cog_window_sync(asset.href, d_bbox)
                if px is not None:
                    r = _analyze_elevation_pixels(px)
                    if r["status"] == "NO-GO":
                        status = "NO-GO"
                    elif r["status"] == "SLOW-GO" and status == "GO":
                        status = "SLOW-GO"
                    factors.append(r["reason"])
                    data_used.append("Copernicus DEM")
                    metrics["elevation"] = r.get("metrics", {})
        except Exception as e:
            logger.error(f"Elevation analysis failed: {e}")

    # Vegetation
    if status == "GO" and terrain_data.get("vegetation_density") and terrain_data["vegetation_density"].get("items"):
        try:
            item = terrain_data["vegetation_density"]["items"][0]
            red_asset = item.assets.get("B04", None)
            nir_asset = item.assets.get("B08", None)
            if red_asset and nir_asset:
                red_px = _read_cog_window_sync(red_asset.href, d_bbox)
                nir_px = _read_cog_window_sync(nir_asset.href, d_bbox)
                if red_px is not None and nir_px is not None:
                    r = _analyze_vegetation_pixels(red_px, nir_px)
                    if r["status"] == "NO-GO":
                        status = "NO-GO"
                    elif r["status"] == "SLOW-GO" and status == "GO":
                        status = "SLOW-GO"
                    factors.append(r["reason"])
                    data_used.append("Sentinel-2 NDVI")
                    metrics["vegetation"] = r.get("metrics", {})
        except Exception as e:
            logger.error(f"Vegetation analysis failed: {e}")

    if not factors:
        factors.append("No major obstructions detected")
        confidence = "low"

    return {
        "direction": direction_name, "cardinal": cardinal,
        "status": status, "factors": factors, "confidence": confidence,
        "data_sources_used": data_used or ["No raster data available"],
        "metrics": metrics
    }


def detect_water_bodies(latitude: float, longitude: float) -> str:
    """Detect water bodies using JRC Global Surface Water occurrence data.
    Uses global water mapping from 1984-2021 to identify permanent and seasonal water.
    Returns water coverage percentage and classification.

    :param latitude: Center latitude of the analysis area
    :param longitude: Center longitude of the analysis area
    :return: JSON string with water detection results
    """
    try:
        logger.info(f"[TOOL] detect_water_bodies at ({latitude:.4f}, {longitude:.4f})")
        bbox = _calculate_bbox(latitude, longitude, RADIUS_MILES)
        items = _query_stac_collection_sync("jrc-gsw", bbox, limit=5)
        if not items:
            return json.dumps({"status": "no_data", "message": "No JRC Global Surface Water data available"})
        asset = items[0].assets.get("occurrence", None)
        if not asset:
            return json.dumps({"status": "no_data", "message": "No water occurrence asset in JRC GSW"})
        px = _read_cog_window_sync(asset.href, bbox)
        if px is None:
            return json.dumps({"status": "error", "message": "Failed to read water occurrence raster"})
        result = _analyze_jrc_water_pixels(px)
        return json.dumps(_convert_numpy_to_python(result))
    except Exception as e:
        logger.error(f"[TOOL] detect_water_bodies failed: {e}")
        return json.dumps({"error": str(e)})


def detect_active_fires(latitude: float, longitude: float) -> str:
    """Detect active fires using MODIS thermal anomaly data.
    Returns fire confidence levels and pixel counts.

    :param latitude: Center latitude of the analysis area
    :param longitude: Center longitude of the analysis area
    :return: JSON string with fire detection results
    """
    try:
        logger.info(f"[TOOL] detect_active_fires at ({latitude:.4f}, {longitude:.4f})")
        bbox = _calculate_bbox(latitude, longitude, RADIUS_MILES)
        items = _query_stac_collection_sync("modis-14A1-061", bbox, limit=5)
        if not items:
            return json.dumps({"status": "no_data", "message": "No MODIS fire data available"})
        asset = items[0].assets.get("FireMask", None)
        if not asset:
            return json.dumps({"status": "no_data", "message": "No FireMask asset"})
        px = _read_cog_window_sync(asset.href, bbox)
        if px is None:
            return json.dumps({"status": "error", "message": "Failed to read fire raster"})
        result = _analyze_fire_pixels(px)
        return json.dumps(_convert_numpy_to_python(result))
    except Exception as e:
        logger.error(f"[TOOL] detect_active_fires failed: {e}")
        return json.dumps({"error": str(e)})


def analyze_slope_for_mobility(latitude: float, longitude: float) -> str:
    """Analyze terrain slope from Copernicus DEM for vehicle mobility.
    Returns slope statistics and GO/SLOW-GO/NO-GO classification.

    :param latitude: Center latitude of the analysis area
    :param longitude: Center longitude of the analysis area
    :return: JSON string with slope analysis and mobility classification
    """
    try:
        logger.info(f"[TOOL] analyze_slope_for_mobility at ({latitude:.4f}, {longitude:.4f})")
        bbox = _calculate_bbox(latitude, longitude, RADIUS_MILES)
        items = _query_stac_collection_sync("cop-dem-glo-30", bbox, limit=5)
        if not items:
            return json.dumps({"status": "no_data", "message": "No DEM data available"})
        asset = items[0].assets.get("data", None)
        if not asset:
            return json.dumps({"status": "no_data", "message": "No DEM data asset"})
        px = _read_cog_window_sync(asset.href, bbox)
        if px is None:
            return json.dumps({"status": "error", "message": "Failed to read DEM raster"})
        result = _analyze_elevation_pixels(px)
        return json.dumps(_convert_numpy_to_python(result))
    except Exception as e:
        logger.error(f"[TOOL] analyze_slope_for_mobility failed: {e}")
        return json.dumps({"error": str(e)})


def analyze_vegetation_density(latitude: float, longitude: float) -> str:
    """Analyze vegetation density using Sentinel-2 NDVI calculation.
    Returns NDVI statistics and vegetation coverage classification.

    :param latitude: Center latitude of the analysis area
    :param longitude: Center longitude of the analysis area
    :return: JSON string with vegetation density analysis
    """
    try:
        logger.info(f"[TOOL] analyze_vegetation_density at ({latitude:.4f}, {longitude:.4f})")
        bbox = _calculate_bbox(latitude, longitude, RADIUS_MILES)
        end_date = datetime.utcnow()
        dt_range = f"{(end_date - timedelta(days=90)).isoformat()}Z/{end_date.isoformat()}Z"
        items = _query_stac_collection_sync("sentinel-2-l2a", bbox, dt_range, query_params={"eo:cloud_cover": {"lt": 50}}, limit=5)
        if not items:
            return json.dumps({"status": "no_data", "message": "No Sentinel-2 data available (may be cloudy)"})
        red_asset = items[0].assets.get("B04", None)
        nir_asset = items[0].assets.get("B08", None)
        if not red_asset or not nir_asset:
            return json.dumps({"status": "no_data", "message": "No Red/NIR band assets"})
        red_px = _read_cog_window_sync(red_asset.href, bbox)
        nir_px = _read_cog_window_sync(nir_asset.href, bbox)
        if red_px is None or nir_px is None:
            return json.dumps({"status": "error", "message": "Failed to read Sentinel-2 raster"})
        result = _analyze_vegetation_pixels(red_px, nir_px)
        return json.dumps(_convert_numpy_to_python(result))
    except Exception as e:
        logger.error(f"[TOOL] analyze_vegetation_density failed: {e}")
        return json.dumps({"error": str(e)})


def create_mobility_functions() -> Set[Callable]:
    """Return the set of mobility tool functions for AsyncFunctionTool registration."""
    return {
        analyze_directional_mobility,
        detect_water_bodies,
        detect_active_fires,
        analyze_slope_for_mobility,
        analyze_vegetation_density,
    }
