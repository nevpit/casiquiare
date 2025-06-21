from agents.context import ContextEngine, context_tools
from tests.test_sample_elevation import (
    _create_dem,
    _create_land_cover,
    _create_soil,
    _create_distance,
    _create_climate,
    _create_context_layer,
)
import pytest
from world_state import reset, world_state


def test_environment_summary_updates_state():
    reset()
    ce = ContextEngine()
    elev = {"elevation": 150.0, "slope_deg": 5.0, "landform": "gentle hill"}
    land = {"category": "forest"}
    soil = {"name": "terra preta", "fertility": "high", "terra_preta": True}
    dist = {"distance_m": 200.0}
    clim = {"mean_annual_precip_mm": 2500.0, "mean_annual_temp_c": 26.0}
    ctx = {"protected_area": False}

    result = ce.environment_summary(elev, land, soil, dist, clim, ctx)
    assert world_state.get("context_environment") == result
    messages = world_state.get("messages", [])
    assert messages
    last = messages[-1]
    assert last["agent"] == "context"
    assert last["type"] == "analysis"


def test_analyze_environment_runtime(tmp_path):
    missing = (
        context_tools.rasterio is None
        or context_tools.np is None
        or context_tools.Transformer is None
        or context_tools.gpd is None
        or context_tools.Point is None
    )
    if missing:
        pytest.skip("geospatial libs required")

    dem = tmp_path / "dem.tif"
    land = tmp_path / "land.tif"
    soil = tmp_path / "soil.tif"
    dist = tmp_path / "dist.tif"
    clim = tmp_path / "clim.tif"
    gpkg = tmp_path / "ctx.gpkg"

    _create_dem(dem)
    _create_land_cover(land)
    _create_soil(soil)
    _create_distance(dist)
    _create_climate(clim)
    _create_context_layer(gpkg)

    ce = ContextEngine()
    reset()
    result = ce.analyze_environment(
        dem=str(dem),
        land_cover=str(land),
        soil=str(soil),
        distance=str(dist),
        climate=str(clim),
        context_layers={"protected_area": str(gpkg)},
        lat=1.0,
        lon=1.0,
    )

    assert result["soil_type"] == "terra preta"
    assert world_state.get("context_environment") == result
