"""Population-aware flood-risk calculations using a WorldPop GeoTIFF."""

import math
import os

import numpy as np
import rasterio
from rasterio.coords import disjoint_bounds
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds as window_from_bounds

POPULATION_PATH = "data/population/ind_pop_2020_CN_1km_R2025A_UA_v1.tif"


def population_density_on_mask(mask_shape, bounds, population_path=POPULATION_PATH):
    """Return WorldPop density (people/km²) aligned to the model mask."""
    if not os.path.exists(population_path):
        return None
    south, west = bounds[0]
    north, east = bounds[1]
    scene_bounds = (west, south, east, north)
    mask_transform = from_bounds(west, south, east, north, mask_shape[1], mask_shape[0])
    with rasterio.open(population_path) as population_source:
        if disjoint_bounds(scene_bounds, population_source.bounds):
            return None
        window = window_from_bounds(*scene_bounds, transform=population_source.transform).round_offsets().round_lengths()
        population = population_source.read(1, window=window, masked=True)
        if population.size == 0:
            return None
        valid_population = np.where(np.ma.getmaskarray(population), 0, population.filled(0))
        valid_population = np.clip(valid_population, 0, None)
        # Convert the source-cell population count to people/km². Source cells
        # use geographic coordinates, so their area changes slightly by row.
        source_rows = window.row_off + np.arange(population.shape[0]) + 0.5
        source_latitudes = population_source.transform.f + source_rows * population_source.transform.e
        source_cell_areas = (
            abs(population_source.transform.a) * 111.320
            * abs(population_source.transform.e) * 110.574
            * np.cos(np.deg2rad(source_latitudes))
        )
        density_source = valid_population / np.maximum(source_cell_areas[:, None], 1e-9)
        density_on_mask = np.zeros(mask_shape, dtype=np.float32)
        reproject(
            source=density_source.astype(np.float32), destination=density_on_mask,
            src_transform=population_source.window_transform(window), src_crs=population_source.crs,
            dst_transform=mask_transform, dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
        )
    return density_on_mask


def population_exposure(flood_mask, bounds, population_density=None, population_path=POPULATION_PATH):
    """Estimate people exposed and density in a flood mask using WorldPop India."""
    if population_density is None:
        population_density = population_density_on_mask(flood_mask.shape, bounds, population_path)
    if population_density is None:
        return None
    if population_density.shape != flood_mask.shape:
        raise ValueError("Population-density grid must match the flood-mask shape.")
    south, west = bounds[0]
    north, east = bounds[1]
    flooded_pixels = flood_mask.astype(bool)
    if not flooded_pixels.any():
        return {"exposed_population": 0, "mean_population_per_km2": 0.0, "density_score": 0.0}
    # Calculate the area of each 10 m-ish model pixel at its latitude, then
    # allocate only that pixel's share of population to the flood mask.
    mask_rows = np.arange(flood_mask.shape[0]) + 0.5
    mask_latitudes = north - mask_rows * (north - south) / flood_mask.shape[0]
    pixel_areas = (
        ((east - west) / flood_mask.shape[1]) * 111.320
        * ((north - south) / flood_mask.shape[0]) * 110.574
        * np.cos(np.deg2rad(mask_latitudes))
    )
    pixel_areas = np.maximum(pixel_areas[:, None], 0)
    flood_area_km2 = float((pixel_areas * flooded_pixels).sum())
    exposed_population = float((population_density * pixel_areas * flooded_pixels).sum())
    mean_population = float(exposed_population / flood_area_km2) if flood_area_km2 else 0.0
    density_score = min(1.0, math.log1p(mean_population) / math.log1p(3000))
    return {"exposed_population": int(round(exposed_population)),
            "mean_population_per_km2": round(mean_population, 1),
            "density_score": round(density_score, 3)}


def calculate_risk(flood_percent, population):
    """Combine flood extent with the number of people exposed (0–1)."""
    flood_score = float(min(1.0, max(0.0, flood_percent / 100.0)))
    # Population density alone is not a flood risk. A scene with no predicted
    # flooded pixels must be reported as no risk, regardless of density.
    if flood_score == 0.0:
        return {"score": 0.0, "level": "no", "flood_score": 0.0}
    exposed_population = max(0, int(population["exposed_population"])) if population else 0
    # Prioritize the people who are actually within the predicted flood area.
    # The logarithmic scale prevents a small difference at high population
    # counts from dominating, while still strongly separating 1 person from
    # 100+ people.
    exposure_score = min(1.0, math.log1p(exposed_population) / math.log1p(100))
    risk_score = float(0.4 * flood_score + 0.6 * exposure_score)
    risk_level = "high" if risk_score >= 0.65 else "medium" if risk_score >= 0.35 else "low"
    return {"score": round(risk_score, 3), "level": risk_level,
            "flood_score": round(flood_score, 3),
            "exposure_score": round(exposure_score, 3)}
