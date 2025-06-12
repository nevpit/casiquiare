# casiquiare
Researching novel methods to help archeological efforts in the Amazon

## Eyes agent utilities

Tools for the Eyes agent live in `eyes/tools.py`. These helpers handle
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

The lower-level `detect_lines` helper extracts straight line segments from a
binary edge image using the Hough transform.

`merge_detections` combines LiDAR and satellite candidates when their
centroids fall within 25 m, helping reduce duplicates across sensors.

Utility modules under `processing` extend the toolkit with low-level I/O helpers.

The `write_geotiff` function in `processing/lidar.py` saves an array to a
GeoTIFF file using a provided rasterio profile so the CRS and transform are
preserved. LiDAR file readers are provided in `io_utils/lidar.py`.

`lidar_tile_dtm`, `lidar_feature_detection` and `scan_lidar_area` can optionally
write their intermediate rasters to disk. Pass ``out_dir`` to specify an output
folder and set ``return_paths=True`` to receive file paths instead of NumPy
arrays.

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

## Programmatic usage examples

Each helper in `eyes/tools.py` can be imported and called directly.  Below
is a minimal example demonstrating all available tools:

```python
from eyes import tools as eyes

# Load point cloud arrays from a LiDAR tile
arrays = eyes.analyze_lidar("tile.laz")

# Inspect basic raster metadata
meta = eyes.analyze_raster("image.tif")

# Analyse Sentinel-2 imagery and compute NDVI
sat = eyes.analyze_satellite_image("sentinel.tif", ndvi_bands=(4, 8))

# Reproject coordinates
x_web, y_web = eyes.transform_coordinates(-60.123, -3.456)

# Detect simple contours in an image
contours = eyes.detect_image_features("hillshade.png")

# Create a DTM and hillshade from LiDAR
dtm_info = eyes.lidar_tile_dtm("tile.laz", out_dir="derived", return_paths=True)

# Run feature detection on a tile
detections = eyes.lidar_feature_detection("tile.laz")

# Use the lower level shape detector
shapes = eyes.detect_shapes(dtm_info["hillshade"], dtm_info["profile"])

# Save 256×256 snippets around features
eyes.save_snippets(dtm_info["hillshade"], detections["features"], "snippets")

# Convenience wrapper for the entire pipeline
results = eyes.scan_lidar_area("tile.laz")
```

## YAML orchestrator snippet

Eyes tools can also be referenced from an external agent orchestrator.  The
example below illustrates how each function could be invoked in a workflow
definition:

```yaml
steps:
  - name: lidar-inspection
    tool: analyze_lidar
    args:
      path: tile.laz

  - name: raster-meta
    tool: analyze_raster
    args:
      path: image.tif

  - name: sentinel-analysis
    tool: analyze_satellite_image
    args:
      path: sentinel.tif
      ndvi_bands: [4, 8]

  - name: to-web-mercator
    tool: transform_coordinates
    args:
      x: -60.123
      y: -3.456
      to_epsg: 3857

  - name: features-from-image
    tool: detect_image_features
    args:
      path: hillshade.png

  - name: create-dtm
    tool: lidar_tile_dtm
    args:
      path: tile.laz
      out_dir: derived
      return_paths: true

  - name: lidar-detection
    tool: lidar_feature_detection
    args:
      path: tile.laz
      size_range: [50, 300]

  - name: raw-detect-shapes
    tool: detect_shapes
    args:
      image: ${hillshade}
      profile: ${profile}

  - name: hough-lines
    tool: detect_lines
    args:
      edge_img: ${edge_mask}

  - name: save-snips
    tool: save_snippets
    args:
      image: ${hillshade}
      features: ${detections}
      out_dir: snippets

  - name: quick-scan
    tool: scan_lidar_area
    args:
      path: tile.laz
```

## Frontend map interface

The React dashboard displays detections on an interactive map. To keep the map
snappy, GeoJSON layers are paginated when more than 2 000 features are loaded.
Use the controls in the bottom-left corner of the map to step through each
page of results.

## Shared world state

The agents share intermediate results through a simple in-memory dictionary
exposed as ``world_state``. The Brain agent records model summaries under
``latest_model`` and cluster information under ``clusters`` while prediction
requests populate ``latest_prediction``.  Each action is logged to a
``messages`` list with ``agent`` and ``type`` fields.  Other agents can import
``world_state`` and inspect these keys to combine findings:

```python
from world_state import world_state
model_info = world_state.get("latest_model")
clusters = world_state.get("clusters", {})
```
