import numpy as np
import pytest

from processing import geo


def test_contour_to_polygon_runtime():
    cnt = np.array([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]], dtype=np.int32)
    transform = geo.Affine(1, 0, 10, 0, -1, 20)
    crs = geo.CRS.from_epsg(4326)

    if geo.rasterio is None:
        with pytest.raises(RuntimeError):
            geo.contour_to_polygon(cnt, None, None)  # type: ignore[arg-type]
    else:
        poly = geo.contour_to_polygon(cnt, transform, crs)
        assert isinstance(poly, list)
        assert len(poly) == 5
        assert poly[0] == poly[-1]
        lon0, lat0 = poly[0]
        assert abs(lon0 - 10.5) < 1e-6
        assert abs(lat0 - 19.5) < 1e-6
        for lon, lat in poly:
            assert lon == round(lon, 6)
            assert lat == round(lat, 6)
