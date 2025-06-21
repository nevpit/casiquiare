"""Utility functions for the Context Engine agent."""

from __future__ import annotations

from typing import Any, Dict

from log_config import setup_logger

try:
    import numpy as np
except Exception:  # pragma: no cover - library may be missing
    np = None  # type: ignore

try:
    import rasterio
    from rasterio.windows import Window
except Exception:  # pragma: no cover - library may be missing
    rasterio = None  # type: ignore
    Window = None  # type: ignore

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - library may be missing
    Transformer = None  # type: ignore

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:  # pragma: no cover - library may be missing
    gpd = None  # type: ignore
    Point = None  # type: ignore

logger = setup_logger("casiquiare.context.tools")

# simple caches for raster and vector data
_RASTER_CACHE: Dict[str, Any] = {}
_VECTOR_CACHE: Dict[str, Any] = {}


def _get_raster(path: str):
    """Return an open raster dataset, caching it for reuse."""
    if rasterio is None:
        raise RuntimeError("rasterio is required")
    if path not in _RASTER_CACHE:
        _RASTER_CACHE[path] = rasterio.open(path)
    return _RASTER_CACHE[path]


def _get_vector(path: str):
    """Return a cached GeoDataFrame."""
    if gpd is None:
        raise RuntimeError("geopandas is required")
    if path not in _VECTOR_CACHE:
        _VECTOR_CACHE[path] = gpd.read_file(path)
    return _VECTOR_CACHE[path]


def clear_cache() -> None:
    """Close cached datasets (for testing or memory management)."""
    for ds in _RASTER_CACHE.values():
        try:
            ds.close()
        except Exception:
            pass
    _RASTER_CACHE.clear()
    _VECTOR_CACHE.clear()



def sample_elevation(path: str, lat: float, lon: float) -> Dict[str, Any]:
    """Sample elevation and terrain metrics at a coordinate from a DEM."""
    if rasterio is None or np is None or Transformer is None:
        raise RuntimeError("rasterio, numpy and pyproj are required.")

    logger.info("Sampling elevation at lat=%s lon=%s from %s", lat, lon, path)
    src = _get_raster(path)
    dem_crs = src.crs
    if dem_crs is None:
        raise ValueError("DEM has no CRS")
    if dem_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(4326, dem_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
    else:
        x, y = lon, lat
    row, col = src.index(x, y)
    window = Window(col - 1, row - 1, 3, 3)
    data = src.read(1, window=window, boundless=True, fill_value=src.nodata)
    center = float(data[1, 1])
    cellsize = src.res[0]
    gy, gx = np.gradient(data.astype(float), cellsize)
    slope = float(np.degrees(np.arctan(np.hypot(gx[1, 1], gy[1, 1]))))
    relief = float(center - data.mean())

    if slope < 3 and abs(relief) < 2:
        landform = "flat floodplain"
    elif slope < 15:
        landform = "gentle hill"
    else:
        landform = "steep ridge"

    result = {
        "elevation": center,
        "slope_deg": slope,
        "local_relief": relief,
        "landform": landform,
    }
    logger.info("sample_elevation result: %s", result)
    return result


def sample_elevation_batch(path: str, coords: "list[tuple[float, float]]") -> "list[dict[str, Any]]":
    """Sample elevation for many coordinates using a cached raster."""
    if rasterio is None or np is None or Transformer is None:
        raise RuntimeError("rasterio, numpy and pyproj are required.")

    src = _get_raster(path)
    dem_crs = src.crs
    if dem_crs is None:
        raise ValueError("DEM has no CRS")
    if dem_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(4326, dem_crs, always_xy=True)
        xy = [transformer.transform(lon, lat) for lat, lon in coords]
    else:
        xy = [(lon, lat) for lat, lon in coords]

    results = []
    for (lon_t, lat_t), (lat, lon) in zip(xy, coords):
        row, col = src.index(lon_t, lat_t)
        window = Window(col - 1, row - 1, 3, 3)
        data = src.read(1, window=window, boundless=True, fill_value=src.nodata)
        center = float(data[1, 1])
        cellsize = src.res[0]
        gy, gx = np.gradient(data.astype(float), cellsize)
        slope = float(np.degrees(np.arctan(np.hypot(gx[1, 1], gy[1, 1]))))
        relief = float(center - data.mean())

        if slope < 3 and abs(relief) < 2:
            landform = "flat floodplain"
        elif slope < 15:
            landform = "gentle hill"
        else:
            landform = "steep ridge"

        results.append(
            {
                "elevation": center,
                "slope_deg": slope,
                "local_relief": relief,
                "landform": landform,
                "lat": lat,
                "lon": lon,
            }
        )

    logger.info("sample_elevation_batch results=%d", len(results))
    return results




def land_cover_class(path: str, lat: float, lon: float) -> Dict[str, Any]:
    """Return land cover code and category for a coordinate."""
    if rasterio is None or Transformer is None:
        raise RuntimeError("rasterio and pyproj are required.")

    logger.info("Sampling land cover at lat=%s lon=%s from %s", lat, lon, path)
    try:
        src = _get_raster(path)
        lcc_crs = src.crs
        if lcc_crs is None:
            raise ValueError("dataset has no CRS")
        if lcc_crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(4326, lcc_crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        samples = list(src.sample([(x, y)]))
        code = int(samples[0][0])
    except rasterio.errors.RasterioIOError:
        if gpd is None or Point is None:
            raise RuntimeError("geopandas is required for vector datasets")
        gdf = _get_vector(path)
        if gdf.crs is None:
            raise ValueError("dataset has no CRS")
        if gdf.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(4326, gdf.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        pt = Point(x, y)
        matches = gdf[gdf.geometry.contains(pt)]
        if matches.empty:
            raise ValueError("no feature found for point")
        code = int(matches.iloc[0][0])

    category = LAND_COVER_CODES.get(code, "unknown")
    result = {"code": code, "category": category}
    logger.info("land_cover_class result: %s", result)
    return result


LAND_COVER_CODES = {
    0: "no data",
    1: "forest",
    2: "savanna",
    3: "agriculture",
    4: "water",
}



def soil_properties(path: str, lat: float, lon: float) -> Dict[str, Any]:
    """Return soil code and attributes for a coordinate."""
    if rasterio is None or Transformer is None:
        raise RuntimeError("rasterio and pyproj are required.")

    logger.info("Sampling soil at lat=%s lon=%s from %s", lat, lon, path)
    try:
        src = _get_raster(path)
        soil_crs = src.crs
        if soil_crs is None:
            raise ValueError("dataset has no CRS")
        if soil_crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(4326, soil_crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        samples = list(src.sample([(x, y)]))
        code = int(samples[0][0])
    except rasterio.errors.RasterioIOError:
        if gpd is None or Point is None:
            raise RuntimeError("geopandas is required for vector datasets")
        gdf = _get_vector(path)
        if gdf.crs is None:
            raise ValueError("dataset has no CRS")
        if gdf.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(4326, gdf.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        pt = Point(x, y)
        matches = gdf[gdf.geometry.contains(pt)]
        if matches.empty:
            raise ValueError("no feature found for point")
        code = int(matches.iloc[0][0])

    attrs = SOIL_CODES.get(code, {"name": "unknown", "fertility": "unknown", "terra_preta": False})
    result = {"code": code, **attrs}
    logger.info("soil_properties result: %s", result)
    return result


SOIL_CODES = {
    0: {"name": "no data", "fertility": "unknown", "terra_preta": False},
    1: {"name": "oxisol", "fertility": "low", "terra_preta": False},
    2: {"name": "terra preta", "fertility": "high", "terra_preta": True},
    3: {"name": "ultisol", "fertility": "moderate", "terra_preta": False},
}


def distance_to_water(path: str, lat: float, lon: float) -> Dict[str, Any]:
    """Return distance in meters to nearest water body for a coordinate."""
    if rasterio is None or Transformer is None:
        raise RuntimeError("rasterio and pyproj are required.")

    logger.info("Sampling distance to water at lat=%s lon=%s from %s", lat, lon, path)
    try:
        src = _get_raster(path)
        dist_crs = src.crs
        if dist_crs is None:
            raise ValueError("dataset has no CRS")
        if dist_crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(4326, dist_crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        samples = list(src.sample([(x, y)]))
        dist = float(samples[0][0])
    except rasterio.errors.RasterioIOError:
        if gpd is None or Point is None:
            raise RuntimeError("geopandas is required for vector datasets")
        gdf = _get_vector(path)
        if gdf.crs is None:
            raise ValueError("dataset has no CRS")
        if gdf.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(4326, gdf.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        pt = Point(x, y)
        dist = float(gdf.distance(pt).min())

    result = {"distance_m": dist}
    logger.info("distance_to_water result: %s", result)
    return result



def climate_variables(path: str, lat: float, lon: float) -> Dict[str, Any]:
    """Return climate variables (precipitation, temperature) for a coordinate."""
    if rasterio is None or Transformer is None:
        raise RuntimeError("rasterio and pyproj are required.")

    logger.info("Sampling climate at lat=%s lon=%s from %s", lat, lon, path)
    src = _get_raster(path)
    clim_crs = src.crs
    if clim_crs is None:
        raise ValueError("dataset has no CRS")
    if clim_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(4326, clim_crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
    else:
        x, y = lon, lat
    samples = list(src.sample([(x, y)]))
    precip = float(samples[0][0])
    temp = float(samples[0][1]) if src.count > 1 else float("nan")

    result = {
        "mean_annual_precip_mm": precip,
        "mean_annual_temp_c": temp,
    }
    logger.info("climate_variables result: %s", result)
    return result




def check_context_layers(layers: Dict[str, str], lat: float, lon: float) -> Dict[str, bool]:
    """Check if a coordinate falls within various context layers."""
    if gpd is None or Point is None or Transformer is None:
        raise RuntimeError("geopandas and pyproj are required.")

    logger.info("Checking context layers at lat=%s lon=%s", lat, lon)
    results: Dict[str, bool] = {}
    for name, path in layers.items():
        gdf = _get_vector(path)
        if gdf.crs is None:
            raise ValueError("dataset has no CRS")
        if gdf.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(4326, gdf.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        pt = Point(x, y)
        results[name] = bool(gdf.contains(pt).any())

    logger.info("check_context_layers result: %s", results)
    return results




def environment_summary(
    elevation: Dict[str, Any],
    land_cover: Dict[str, Any],
    soil: Dict[str, Any],
    distance: Dict[str, Any],
    climate: Dict[str, Any],
    context_layers: Dict[str, bool],
) -> Dict[str, Any]:
    """Combine metrics into a structured summary and compute suitability."""

    logger.info("Compiling environment summary")

    summary = {
        "elevation_m": elevation.get("elevation"),
        "slope_deg": elevation.get("slope_deg"),
        "landform": elevation.get("landform"),
        "land_cover": land_cover.get("category"),
        "soil_type": soil.get("name"),
        "soil_fertility": soil.get("fertility"),
        "terra_preta": soil.get("terra_preta"),
        "distance_to_water_m": distance.get("distance_m"),
        "mean_annual_precip_mm": climate.get("mean_annual_precip_mm"),
        "mean_annual_temp_c": climate.get("mean_annual_temp_c"),
    }
    summary.update(context_layers)

    score = 0.0

    fertility = soil.get("fertility")
    if fertility == "high":
        score += 1.0
    elif fertility == "moderate":
        score += 0.5

    dist = distance.get("distance_m")
    if dist is not None:
        if dist <= 1000:
            score += 1.0
        elif dist <= 5000:
            score += 0.5
        else:
            score -= 0.5

    landform = elevation.get("landform")
    if landform == "gentle hill":
        score += 1.0
    elif landform == "flat floodplain":
        score += 0.5
    else:
        score -= 0.5

    precip = climate.get("mean_annual_precip_mm")
    if precip is not None:
        if 500 <= precip <= 3500:
            score += 0.5
        else:
            score -= 0.5

    if context_layers.get("protected_area"):
        score -= 1.0

    summary["suitability_score"] = score

    logger.info("environment_summary result: %s", summary)
    return summary




def _historical_clue_note(lat: float, lon: float, summary: Dict[str, Any]) -> str:
    """Return a note comparing historical clues with environment data."""

    from world_state import world_state

    key = f"{lon:.4f},{lat:.4f}"
    clue = world_state.get("historical_clues", {}).get(key)
    if not clue:
        return ""

    clue_lower = str(clue).lower()
    dist = summary.get("distance_to_water_m")

    if "lake" in clue_lower:
        if dist is not None and dist <= 500:
            note = (
                f"Historical clue: {clue}. Environmental data shows nearby water, "
                "supporting the account."
            )
        else:
            note = (
                f"Historical clue: {clue}. No lake is evident nearby, though this "
                "does not rule out a past water body."
            )
    elif "river" in clue_lower:
        if dist is not None and dist <= 5000:
            note = (
                f"Historical clue: {clue}. A nearby river aligns with this detail."
            )
        else:
            note = (
                f"Historical clue: {clue}. However, the closest major river seems "
                "several kilometres away."
            )
    else:
        note = f"Historical clue: {clue}."

    logger.info("historical_clue_note result: %s", note)
    return note


def generate_context_summary(
    summary: Dict[str, Any], lat: float | None = None, lon: float | None = None
) -> str:
    """Craft a human-readable narrative from :func:`environment_summary`."""

    logger.info("Generating context narrative")

    parts = []

    elev = summary.get("elevation_m")
    landform = summary.get("landform")
    if elev is not None and landform:
        parts.append(f"Elevation {elev:.0f} m on a {landform}")
    elif elev is not None:
        parts.append(f"Elevation {elev:.0f} m")

    soil = summary.get("soil_type")
    fert = summary.get("soil_fertility")
    if soil:
        desc = soil
        if fert and fert != "unknown":
            desc = f"{fert} fertility {soil}"
        if summary.get("terra_preta"):
            desc = f"nutrient-rich {soil}"
        parts.append(f"with {desc} soil")

    cover = summary.get("land_cover")
    if cover:
        parts.append(f"land cover is {cover}")

    dist = summary.get("distance_to_water_m")
    if dist is not None:
        if dist >= 1000:
            parts.append(f"about {dist/1000:.1f} km from water")
        else:
            parts.append(f"{dist:.0f} m from water")

    precip = summary.get("mean_annual_precip_mm")
    temp = summary.get("mean_annual_temp_c")
    if precip is not None and temp is not None:
        parts.append(f"annual rainfall ~{precip:.0f} mm with mean temp {temp:.1f} °C")

    flags = [name.replace("_", " ") for name, val in summary.items() if isinstance(val, bool) and val]
    if flags:
        parts.append(f"within {' and '.join(flags)}")

    narrative = "; ".join(parts) + "."

    if lat is not None and lon is not None:
        note = _historical_clue_note(lat, lon, summary)
        if note:
            narrative += " " + note

    logger.info("generate_context_summary result: %s", narrative)
    return narrative


TOOLS: Dict[str, Any] = {
    "sample_elevation": sample_elevation,
    "sample_elevation_batch": sample_elevation_batch,
    "land_cover_class": land_cover_class,
    "soil_properties": soil_properties,
    "distance_to_water": distance_to_water,
    "climate_variables": climate_variables,
    "check_context_layers": check_context_layers,
    "environment_summary": environment_summary,
    "generate_context_summary": generate_context_summary,
}


__all__ = [
    "sample_elevation",
    "sample_elevation_batch",
    "land_cover_class",
    "soil_properties",
    "distance_to_water",
    "climate_variables",
    "check_context_layers",
    "environment_summary",
    "generate_context_summary",
    "clear_cache",
    "TOOLS",
]
