import pytest

from processing import geo


def test_pixel_to_coords_runtime():
    if geo.rasterio is None:
        with pytest.raises(RuntimeError):
            geo.pixel_to_coords(0, 0, None, None)  # type: ignore[arg-type]
    else:
        transform = geo.Affine(1, 0, 10, 0, -1, 20)
        crs = geo.CRS.from_epsg(4326)
        lon, lat = geo.pixel_to_coords(0, 0, transform, crs)
        assert abs(lon - 10.5) < 1e-6
        assert abs(lat - 19.5) < 1e-6
        assert lon == round(lon, 6)
        assert lat == round(lat, 6)

