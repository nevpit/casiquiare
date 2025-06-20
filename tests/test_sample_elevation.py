import numpy as np
import pytest

from world_state import reset, world_state

from agents.context import context_tools


def _create_dem(path):
    import rasterio
    from rasterio.transform import from_origin

    data = np.arange(25, dtype=np.float32).reshape((5, 5))
    transform = from_origin(0, 5, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def _create_land_cover(path):
    import rasterio
    from rasterio.transform import from_origin

    data = np.array([
        [1, 2, 2],
        [1, 3, 3],
        [0, 0, 4],
    ], dtype=np.uint8)
    transform = from_origin(0, 3, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def _create_soil(path):
    import rasterio
    from rasterio.transform import from_origin

    data = np.array([
        [1, 2, 0],
        [3, 2, 1],
    ], dtype=np.uint8)
    transform = from_origin(0, 2, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def _create_distance(path):
    import rasterio
    from rasterio.transform import from_origin

    data = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ], dtype=np.float32)
    transform = from_origin(0, 3, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def _create_climate(path):
    import rasterio
    from rasterio.transform import from_origin

    precip = np.array([[1000.0, 1100.0], [900.0, 950.0]], dtype=np.float32)
    temp = np.array([[25.0, 26.0], [24.0, 23.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(precip, 1)
        dst.write(temp, 2)


def _create_context_layer(path):
    import geopandas as gpd
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame(
        {"name": ["park"]},
        geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GPKG")


def test_sample_elevation_runtime(tmp_path):
    tif = tmp_path / "dem.tif"
    _create_dem(tif)

    missing = (
        context_tools.rasterio is None
        or context_tools.np is None
        or context_tools.Transformer is None
    )

    if missing:
        with pytest.raises(RuntimeError):
            context_tools.sample_elevation(str(tif), 4.0, 1.0)
    else:
        res = context_tools.sample_elevation(str(tif), 4.0, 1.0)
        assert "elevation" in res
        assert res["elevation"] == 6.0
        assert "landform" in res


def test_land_cover_class_runtime(tmp_path):
    tif = tmp_path / "landcover.tif"
    _create_land_cover(tif)

    missing = context_tools.rasterio is None or context_tools.Transformer is None

    if missing:
        with pytest.raises(RuntimeError):
            context_tools.land_cover_class(str(tif), 1.5, 1.5)
    else:
        res = context_tools.land_cover_class(str(tif), 1.5, 1.5)
        assert res["code"] == 3
        assert res["category"] == "agriculture"


def test_soil_properties_runtime(tmp_path):
    tif = tmp_path / "soil.tif"
    _create_soil(tif)

    missing = context_tools.rasterio is None or context_tools.Transformer is None

    if missing:
        with pytest.raises(RuntimeError):
            context_tools.soil_properties(str(tif), 0.5, 1.5)
    else:
        res = context_tools.soil_properties(str(tif), 0.5, 1.5)
        assert res["code"] == 2
        assert res["name"] == "terra preta"
        assert res["terra_preta"] is True


def test_distance_to_water_runtime(tmp_path):
    tif = tmp_path / "dist.tif"
    _create_distance(tif)

    missing = context_tools.rasterio is None or context_tools.Transformer is None

    if missing:
        with pytest.raises(RuntimeError):
            context_tools.distance_to_water(str(tif), 1.5, 1.5)
    else:
        res = context_tools.distance_to_water(str(tif), 1.5, 1.5)
        assert res["distance_m"] == 2.0


def test_climate_variables_runtime(tmp_path):
    tif = tmp_path / "climate.tif"
    _create_climate(tif)

    missing = context_tools.rasterio is None or context_tools.Transformer is None

    if missing:
        with pytest.raises(RuntimeError):
            context_tools.climate_variables(str(tif), 0.5, 0.5)
    else:
        res = context_tools.climate_variables(str(tif), 1.5, 0.5)
        assert res["mean_annual_precip_mm"] == 1000.0
        assert res["mean_annual_temp_c"] == 25.0


def test_check_context_layers_runtime(tmp_path):
    missing = (
        context_tools.gpd is None
        or context_tools.Point is None
        or context_tools.Transformer is None
    )

    gpkg = tmp_path / "context.gpkg"
    if not missing:
        _create_context_layer(gpkg)

    layers = {"protected_area": str(gpkg)}
    if missing:
        with pytest.raises(RuntimeError):
            context_tools.check_context_layers(layers, 1.0, 1.0)
    else:
        inside = context_tools.check_context_layers(layers, 1.0, 1.0)
        outside = context_tools.check_context_layers(layers, 3.0, 3.0)
        assert inside["protected_area"] is True
        assert outside["protected_area"] is False


def test_environment_summary_runtime(tmp_path):
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

    elev = context_tools.sample_elevation(str(dem), 4.0, 1.0)
    lc = context_tools.land_cover_class(str(land), 1.5, 1.5)
    soil_res = context_tools.soil_properties(str(soil), 0.5, 1.5)
    dist_res = context_tools.distance_to_water(str(dist), 1.5, 1.5)
    clim_res = context_tools.climate_variables(str(clim), 1.5, 0.5)
    ctx = context_tools.check_context_layers({"protected_area": str(gpkg)}, 1.0, 1.0)

    summary = context_tools.environment_summary(
        elev, lc, soil_res, dist_res, clim_res, ctx
    )

    assert summary["soil_type"] == "terra preta"
    assert summary["land_cover"] == "agriculture"
    assert summary["protected_area"] is True
    assert summary["suitability_score"] >= 1.0

    text = context_tools.generate_context_summary(summary)
    assert "terra preta" in text
    assert "agriculture" in text


def test_raster_cache(tmp_path, monkeypatch):
    missing = context_tools.rasterio is None or context_tools.np is None or context_tools.Transformer is None
    if missing:
        pytest.skip("geospatial libs required")

    tif = tmp_path / "cache_dem.tif"
    _create_dem(tif)

    open_calls = []

    real_open = context_tools.rasterio.open

    def counting_open(path, *args, **kwargs):
        open_calls.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(context_tools.rasterio, "open", counting_open)
    context_tools.clear_cache()
    context_tools.sample_elevation(str(tif), 1.0, 1.0)
    context_tools.sample_elevation(str(tif), 2.0, 2.0)
    assert open_calls.count(str(tif)) == 1


def test_sample_elevation_batch(tmp_path):
    missing = (
        context_tools.rasterio is None
        or context_tools.np is None
        or context_tools.Transformer is None
    )
    if missing:
        pytest.skip("geospatial libs required")

    tif = tmp_path / "batch_dem.tif"
    _create_dem(tif)
    context_tools.clear_cache()
    results = context_tools.sample_elevation_batch(
        str(tif), [(0.5, 0.5), (1.5, 1.5)]
    )
    assert len(results) == 2
    assert results[0]["elevation"] == 20.0


def test_generate_context_summary_with_clue(tmp_path):
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

    elev = context_tools.sample_elevation(str(dem), 4.0, 1.0)
    lc = context_tools.land_cover_class(str(land), 1.5, 1.5)
    soil_res = context_tools.soil_properties(str(soil), 0.5, 1.5)
    dist_res = context_tools.distance_to_water(str(dist), 1.5, 1.5)
    clim_res = context_tools.climate_variables(str(clim), 1.5, 0.5)
    ctx = context_tools.check_context_layers({"protected_area": str(gpkg)}, 1.0, 1.0)

    summary = context_tools.environment_summary(
        elev, lc, soil_res, dist_res, clim_res, ctx
    )

    reset()
    key = "1.0000,1.0000"
    world_state["historical_clues"] = {key: "legends say a large lake once existed"}

    text = context_tools.generate_context_summary(summary, lat=1.0, lon=1.0)
    assert "Historical clue" in text
    assert "lake" in text

