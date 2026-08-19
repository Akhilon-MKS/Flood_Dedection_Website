"""Standalone web dashboard. Run with: python app.py"""
import base64
import os
import tempfile

import cv2
import numpy as np
import rasterio
from flask import Flask, jsonify, render_template, request
from rasterio.warp import transform_bounds
from werkzeug.utils import secure_filename

from infer import run_inference
from risk import calculate_risk, population_exposure

MODEL_PATH = "models/flood_unet.pth"
ALLOWED_EXTENSIONS = {"tif", "tiff"}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_bounds(image_path):
    with rasterio.open(image_path) as source:
        if source.count < 2:
            raise ValueError("The GeoTIFF must contain at least two SAR bands (VV and VH).")
        bounds = source.bounds
        if source.crs and source.crs.to_string() != "EPSG:4326":
            west, south, east, north = transform_bounds(source.crs, "EPSG:4326", *bounds)
        elif source.crs:
            west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
        else:
            raise ValueError("The uploaded GeoTIFF has no coordinate reference system (CRS).")
    return [[south, west], [north, east]]


def mask_to_data_url(mask):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 239, 68, 68
    rgba[..., 3] = (mask * 165).astype(np.uint8)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not success:
        raise ValueError("Could not create the flood-mask overlay.")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def add_tile_bounds(tiles, scene_bounds, mask_shape, tile_size=32):
    """Attach latitude/longitude bounds to each ranked model tile."""
    south, west = scene_bounds[0]
    north, east = scene_bounds[1]
    height, width = mask_shape
    lat_span, lon_span = north - south, east - west
    for tile in tiles:
        row_start, col_start = (tile["row"] - 1) * tile_size, (tile["column"] - 1) * tile_size
        row_end, col_end = min(row_start + tile_size, height), min(col_start + tile_size, width)
        tile_north = north - (row_start / height) * lat_span
        tile_south = north - (row_end / height) * lat_span
        tile_west = west + (col_start / width) * lon_span
        tile_east = west + (col_end / width) * lon_span
        tile["bounds"] = [[tile_south, tile_west], [tile_north, tile_east]]
    return tiles


def add_tile_metrics(tiles, mask, scene_bounds, tile_size=32):
    """Add local flood, population, and risk estimates to ranked tiles."""
    height, width = mask.shape
    for tile in tiles:
        row_start, col_start = (tile["row"] - 1) * tile_size, (tile["column"] - 1) * tile_size
        row_end, col_end = min(row_start + tile_size, height), min(col_start + tile_size, width)
        tile_mask = mask[row_start:row_end, col_start:col_end]
        flooded_pixels = int(tile_mask.sum())
        # The model mask represents 10 m × 10 m Sentinel-1 ground pixels.
        area_km2 = flooded_pixels * (10 ** 2) / 1_000_000
        flood_percent = float(tile_mask.mean() * 100) if tile_mask.size else 0.0
        population = population_exposure(tile_mask, tile["bounds"])
        risk = calculate_risk(flood_percent, population)
        tile["area_km2"] = round(area_km2, 3)
        tile["flooded_pixels"] = flooded_pixels
        tile["flood_percent"] = round(flood_percent, 1)
        tile["population"] = population
        tile["risk"] = risk
    return tiles


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", population_risk_enabled=True)


@app.post("/api/analyze")
def analyze():
    upload = request.files.get("image")
    if not upload or not upload.filename:
        return jsonify(error="Choose a Sentinel-1 GeoTIFF before running analysis."), 400
    if not allowed_file(upload.filename):
        return jsonify(error="Only .tif and .tiff files are supported."), 400
    if not os.path.exists(MODEL_PATH):
        return jsonify(error=f"Model checkpoint not found: {MODEL_PATH}"), 500
    try:
        threshold = float(request.form.get("threshold", 0.5))
        if not 0.1 <= threshold <= 0.9:
            raise ValueError("Threshold must be between 0.10 and 0.90.")
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    suffix = os.path.splitext(secure_filename(upload.filename))[1].lower() or ".tif"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            upload.save(temp_file)
        bounds = image_bounds(temp_path)
        result = run_inference(temp_path, MODEL_PATH, threshold=threshold)
        flood_percent = float(result["mask"].mean() * 100)
        population = population_exposure(result["mask"], bounds)
        risk = calculate_risk(flood_percent, population)
        priority_tiles = add_tile_bounds(result["priority_tiles"], bounds, result["mask"].shape)
        priority_tiles = add_tile_metrics(priority_tiles, result["mask"], bounds)
        return jsonify(area_km2=result["area_km2"], flooded_pixels=result["flooded_pixels"],
                       flood_percent=round(flood_percent, 1), risk=risk, population=population,
                       bounds=bounds, mask_url=mask_to_data_url(result["mask"]),
                       priority_tiles=priority_tiles)
    except Exception as exc:
        app.logger.exception("Analysis failed")
        return jsonify(error=f"Unable to process this image: {exc}"), 422
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify(error="File is too large. The maximum upload size is 200 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
