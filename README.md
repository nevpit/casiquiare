# casiquiare
Researching novel methods to help archeological efforts in the Amazon

## Eyes agent utilities

Tools for the Eyes agent live in `agents/eyes/eyes_tools.py`. These helpers handle
LiDAR processing, raster inspection, coordinate transforms, and shape
detection. They are callable directly or via `Eyes.tools` and wrapped as
methods on the `Eyes` agent. The shape detection logic now resides in the
`detection` package and is re-exported for convenience.

The agent is instructed with a system prompt defining its persona as the
Remote-Sensing & GIS Lead. Any natural-language summary returned by the Eyes
agent is formatted as JSON:

```json
{"agent": "eyes", "type": "summary", "content": "<text>"}
```
and the JSON is wrapped inside a single markdown block.

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
`fuse_score_detections` builds on this by merging detections and immediately
scoring them with a trained model.

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

Each helper in `agents/eyes/eyes_tools.py` can be imported and called directly.  Below
is a minimal example demonstrating all available tools:

```python
from agents.eyes import eyes_tools as eyes

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

## Configurable Brain models

The ``Brain`` agent trains a classifier to estimate site likelihoods. By
default a ``RandomForestClassifier`` is used, but ``train_model`` now accepts a
``model_type`` option so you can switch to ``xgboost`` if the ``xgboost``
package is installed:

```python
from agents.brain import brain_tools

info = brain_tools.train_model(df, model_type="xgboost", n_estimators=200)
```

## Milvus vector database connection

The project can optionally connect to a self-hosted Milvus instance using the
``pymilvus`` client. Connection parameters are read from the ``MILVUS_HOST`` and
``MILVUS_PORT`` environment variables and default to ``localhost`` and
``19530`` respectively. The helper :func:`connect_milvus` in ``milvus_client``
establishes the connection:

```python
from milvus_client import connect_milvus

conn = connect_milvus()  # uses environment variables if provided
```

The Flask backend automatically attempts this connection when
``create_app`` is called, storing the handle under ``app.extensions['milvus_conn']``.

To store text embeddings, a helper is provided to create a collection with a
1536-dimensional vector field and metadata columns using the L2 metric.  The
``index_documents`` function now writes each text chunk's embedding to this
collection along with its ``doc_id``, ``title`` and page number instead of
building an in-memory FAISS index:

```python
from milvus_client import create_embeddings_collection

collection = create_embeddings_collection()
```

The helper :func:`create_image_embeddings_collection` sets up a separate
collection for storing CLIP image vectors with a ``512``-dimensional
``embedding`` field and an ``image_id`` metadata column:

```python
from milvus_client import create_image_embeddings_collection

img_coll = create_image_embeddings_collection()
```

