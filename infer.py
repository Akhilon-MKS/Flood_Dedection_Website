"""
infer.py
--------
Run the trained flood-segmentation model on a new Sentinel-1 SAR GeoTIFF
(e.g. a Tamil Nadu image you pulled from Google Earth Engine / Copernicus)
and return the predicted flood mask plus estimated flooded area.

Can be used standalone:
    python infer.py --image path/to/tn_image.tif --model models/flood_unet.pth

Or imported into the Streamlit app (see app.py).
"""

import argparse
import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp

from dataset import normalize_sar

PIXEL_RESOLUTION_M = 10  # Sentinel-1 ground resolution


def load_model(model_path, device):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # weights come from checkpoint, not ImageNet
        in_channels=2,
        classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(path, img_size=256):
    """Load a 2-band (VV, VH) SAR GeoTIFF and prepare it for the model."""
    with rasterio.open(path) as src:
        arr = src.read()               # (2, H, W)
        transform = src.transform
        crs = src.crs

    arr = np.transpose(arr, (1, 2, 0))  # (H, W, 2)
    arr = normalize_sar(arr).astype(np.float32)

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
    tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear")

    return tensor, transform, crs, arr.shape[:2]


def predict_flood_mask(model, image_tensor, device, threshold=0.5):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > threshold).float().cpu().numpy()[0, 0]  # (H, W)
        prob_map = prob.cpu().numpy()[0, 0]
    return mask, prob_map


def estimate_area(mask, pixel_res_m=PIXEL_RESOLUTION_M):
    """Convert a binary mask into flooded area in km^2."""
    flooded_pixels = int(mask.sum())
    area_km2 = flooded_pixels * (pixel_res_m ** 2) / 1_000_000
    return flooded_pixels, area_km2


def rank_priority_tiles(prob_map, tile_size=32, threshold=0.5, top_k=None):
    """Return model tiles, optionally limited by flood coverage and confidence."""
    height, width = prob_map.shape
    tiles = []
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            tile = prob_map[row:min(row + tile_size, height), col:min(col + tile_size, width)]
            coverage = float((tile >= threshold).mean())
            confidence = float(tile.mean())
            tiles.append({
                "row": row // tile_size + 1,
                "column": col // tile_size + 1,
                "flood_coverage": coverage,
                "confidence": confidence,
                "priority_score": coverage * confidence,
            })
    ranked_tiles = sorted(tiles, key=lambda item: item["priority_score"], reverse=True)
    return ranked_tiles[:top_k] if top_k is not None else ranked_tiles


def run_inference(image_path, model_path, device=None, img_size=256, threshold=0.5):
    """Convenience wrapper: image path in, results dict out."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    tensor, transform, crs, orig_shape = preprocess_image(image_path, img_size)
    mask, prob_map = predict_flood_mask(model, tensor, device, threshold)
    flooded_pixels, area_km2 = estimate_area(mask)
    # The web dashboard applies the final population-aware ranking after it
    # enriches every tile with local exposure and risk information.
    priority_tiles = rank_priority_tiles(prob_map, threshold=threshold, top_k=None)

    return {
        "mask": mask,              # (img_size, img_size) binary array
        "prob_map": prob_map,      # (img_size, img_size) probability array
        "flooded_pixels": flooded_pixels,
        "area_km2": round(area_km2, 3),
        "transform": transform,
        "crs": crs,
        "orig_shape": orig_shape,
        "priority_tiles": priority_tiles,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path to SAR GeoTIFF (2-band VV/VH)")
    parser.add_argument("--model", default="models/flood_unet.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = run_inference(args.image, args.model, threshold=args.threshold)
    print(f"Flooded pixels: {result['flooded_pixels']}")
    print(f"Estimated flooded area: {result['area_km2']} km²")
