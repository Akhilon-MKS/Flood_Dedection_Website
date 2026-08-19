"""
app.py
------
Streamlit dashboard: upload a Sentinel-1 SAR image (GeoTIFF), run the
trained flood-segmentation model, and view the predicted flood extent
overlaid on a Tamil Nadu map.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import transform_bounds
import tempfile
import os

from infer import run_inference

st.set_page_config(page_title="Flood Detection - Tamil Nadu", layout="wide")

st.title("🌊 Satellite-Based Flood Detection — Tamil Nadu")
st.markdown(
    "Upload a Sentinel-1 SAR image (2-band GeoTIFF: VV, VH) of a Tamil Nadu "
    "region to detect flooded areas and estimate affected area."
)

MODEL_PATH = "models/flood_unet.pth"

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Flood probability threshold", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.markdown(
        "**Model:** U-Net (ResNet34 encoder)\n\n"
        "**Trained on:** Sen1Floods11\n\n"
        "**Input:** Sentinel-1 SAR (VV + VH)"
    )

uploaded_file = st.file_uploader("Upload Sentinel-1 SAR GeoTIFF", type=["tif", "tiff"])

# Tamil Nadu approximate center for default map view
TN_CENTER = [11.1271, 78.6569]

if uploaded_file is not None:
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"No trained model found at `{MODEL_PATH}`. "
            "Run `python train.py --data_dir <path>` first to produce it."
        )
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Running flood detection model..."):
            try:
                result = run_inference(tmp_path, MODEL_PATH, threshold=threshold)

                # get lat/lon bounds of the uploaded image for map placement
                with rasterio.open(tmp_path) as src:
                    bounds = src.bounds
                    if src.crs and src.crs.to_string() != "EPSG:4326":
                        west, south, east, north = transform_bounds(
                            src.crs, "EPSG:4326", *bounds
                        )
                    else:
                        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("Predicted Flood Map")
                    m = folium.Map(
                        location=[(south + north) / 2, (west + east) / 2],
                        zoom_start=11,
                        tiles="OpenStreetMap",
                    )

                    # overlay predicted mask as a red-tinted image layer
                    mask_rgba = np.zeros((*result["mask"].shape, 4), dtype=np.uint8)
                    mask_rgba[..., 0] = 255  # red channel
                    mask_rgba[..., 3] = (result["mask"] * 150).astype(np.uint8)  # alpha

                    folium.raster_layers.ImageOverlay(
                        image=mask_rgba,
                        bounds=[[south, west], [north, east]],
                        opacity=0.7,
                        name="Predicted Flood Extent",
                    ).add_to(m)

                    folium.LayerControl().add_to(m)
                    st_folium(m, width=800, height=550)

                with col2:
                    st.subheader("Flood Impact Summary")
                    st.metric("Estimated Flooded Area", f"{result['area_km2']} km²")
                    st.metric("Flooded Pixels", f"{result['flooded_pixels']:,}")

                    flood_pct = (result["mask"].sum() / result["mask"].size) * 100
                    st.metric("% of Image Flooded", f"{flood_pct:.1f}%")

                    st.subheader("High-Priority Flood Zones")
                    st.caption("Tiles are ranked by predicted flood coverage and confidence.")
                    for rank, tile in enumerate(result["priority_tiles"], start=1):
                        st.write(
                            f"{rank}. Row {tile['row']}, column {tile['column']} | "
                            f"{tile['flood_coverage'] * 100:.1f}% flooded | "
                            f"priority {tile['priority_score']:.3f}"
                        )

                    if flood_pct > 30:
                        st.error("🔴 HIGH severity — large-scale flooding detected")
                    elif flood_pct > 10:
                        st.warning("🟠 MEDIUM severity — moderate flooding detected")
                    else:
                        st.success("🟢 LOW severity — limited flooding detected")

                    st.markdown("---")
                    st.caption(
                        "Note: severity thresholds above are illustrative for the MVP. "
                        "Production use should incorporate population and infrastructure "
                        "overlays for accurate rescue-priority ranking."
                    )

            except Exception as e:
                st.error(f"Error processing image: {e}")
            finally:
                os.unlink(tmp_path)

else:
    st.info("👆 Upload a Sentinel-1 SAR GeoTIFF to get started.")
    st.subheader("Tamil Nadu — Default View")
    m = folium.Map(location=TN_CENTER, zoom_start=7, tiles="OpenStreetMap")
    st_folium(m, width=1000, height=500)
