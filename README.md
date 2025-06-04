# casiquiare
Researching novel methods to help archeological efforts in the Amazon

## Eyes agent utilities

Tools for the Eyes agent live in `agents/eyes_tools.py`. These helpers handle
LiDAR processing, raster inspection, coordinate transforms, and shape
detection. They are callable directly or via `Eyes.tools` and wrapped as
methods on the `Eyes` agent. The shape detection logic now resides in the
`detection` package and is re-exported for convenience.

New tools focus on identifying vegetation or soil anomalies with Sentinel-2
multi-spectral imagery. The helper `io_utils.raster.load_raster` loads these rasters
with their coordinate metadata intact for downstream analysis. `analyze_satellite_image`
optionally computes spectral indices like NDVI and NDWI for vegetation-anomaly
detection.

The `detect_shapes` function scans a hillshade or local relief image for
geometric forms and reports their approximate size. Supplying the raster
profile allows the results to be expressed in meters so features outside the
50–300 m range can be filtered out. The dilation kernel used when
extracting edges and the contour size limits are exposed as parameters so
you can tune detection sensitivity to each dataset.

`merge_detections` combines LiDAR and satellite candidates when their
centroids fall within 25 m, helping reduce duplicates across sensors.

Utility modules under `processing` extend the toolkit with low-level I/O helpers.

The `write_geotiff` function in `processing/lidar.py` saves an array to a
GeoTIFF file using a provided rasterio profile so the CRS and transform are
preserved. LiDAR file readers are provided in `io_helpers/lidar.py`.

Additional helpers such as `load_laz`, `generate_lrm`, `generate_hillshade`, and
`build_dtm` expose low-level I/O and terrain modelling capabilities through the
Eyes toolkit.

Use `save_snippets` to crop 256 × 256 PNG images around features returned
by `detect_shapes` for quick visual inspection. Detection results can be
saved with `output.serialize_features` or exported as a GeoJSON
`FeatureCollection` using `output.save_geojson`.

`scan_tiles_concurrent` leverages `concurrent.futures` to run `scan_area` on
multiple raster tiles in parallel, returning the detections in input order.

## Optional dependencies

Some advanced features, such as generating hillshades and local relief models,
rely on `scipy` and the `gdal` library (available as the `osgeo` module). Install
these packages alongside the standard requirements if you need the full
functionality of the processing utilities.
