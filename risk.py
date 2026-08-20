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


def population_exposure(flood_mask, bounds, population_path=POPULATION_PATH):
    """Estimate people exposed and density in a flood mask using WorldPop India."""
    if not os.path.exists(population_path):
        return None
    south, west = bounds[0]
    north, east = bounds[1]
    scene_bounds = (west, south, east, north)
    mask_transform = from_bounds(west, south, east, north, flood_mask.shape[1], flood_mask.shape[0])
    with rasterio.open(population_path) as population_source:
        if disjoint_bounds(scene_bounds, population_source.bounds):
            return None
        window = window_from_bounds(*scene_bounds, transform=population_source.transform).round_offsets().round_lengths()
        population = population_source.read(1, window=window, masked=True)
        if population.size == 0:
            return None
        # A 1 km WorldPop cell is much larger than an individual flood-model
        # pixel. Average coverage keeps partial flooding rather than sampling
        # only one mask pixel at the population-cell centre.
        flood_on_population_grid = np.zeros(population.shape, dtype=np.float32)
        reproject(
            source=flood_mask.astype(np.float32), destination=flood_on_population_grid,
            src_transform=mask_transform, src_crs="EPSG:4326",
            dst_transform=population_source.window_transform(window), dst_crs=population_source.crs,
            resampling=Resampling.average,
        )
    valid_population = np.where(np.ma.getmaskarray(population), 0, population.filled(0))
    valid_population = np.clip(valid_population, 0, None)
    flood_fraction = np.clip(flood_on_population_grid, 0, 1)
    flooded_cells = flood_fraction > 0
    if not flooded_cells.any():
        return {"exposed_population": 0, "mean_population_per_km2": 0.0, "density_score": 0.0}
    # Population is weighted by the predicted flooded fraction of each cell.
    exposed_population = float((valid_population * flood_fraction).sum())
    mean_population = float(exposed_population / flood_fraction.sum())
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
