# casiquiare
Researching novel methods to help archeological efforts in the Amazon

## Eyes agent utilities

Tools for the Eyes agent live in `agents/eyes_tools.py`. These helpers handle
LiDAR processing, raster inspection, coordinate transforms, and shape
detection. They are callable directly or via `Eyes.tools` and wrapped as
methods on the `Eyes` agent. The shape detection logic now resides in the
`detection` package and is re-exported for convenience.

The `detect_shapes` function scans a hillshade or local relief image for
geometric forms and reports their approximate size. Supplying the raster
profile allows the results to be expressed in meters so features outside the
50–300 m range can be filtered out.

Utility modules under `processing` extend the toolkit with low-level I/O helpers.
The `write_geotiff` function in `processing/lidar.py` saves an array to a
GeoTIFF file using a provided rasterio profile so the CRS and transform are
preserved.

## Optional dependencies

Some advanced features, such as generating hillshades and local relief models,
rely on `scipy` and the `gdal` library (available as the `osgeo` module). Install
these packages alongside the standard requirements if you need the full
functionality of the processing utilities.

## CLI usage

Run scans manually with the ``eyes-agent`` helper script. The basic syntax is::

    eyes-agent scan <area_id> [--save-geojson] [--save-snippets]

``area_id`` must exist in ``config/data_paths.yaml``. Use the optional flags to
export detected features as a GeoJSON file and to save small PNG snippets around
each hit.
