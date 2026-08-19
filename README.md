# Satellite Flood Detection — Tamil Nadu (Hackathon MVP)

U-Net (ResNet34 encoder) flood segmentation model trained on Sen1Floods11,
with a Streamlit dashboard showing predicted flood extent on a Tamil Nadu map.

## Verified Working
All code in this project has been syntax-checked and the model architecture
has been tested with a forward pass (2-channel SAR input -> flood mask output).
Loss functions and IoU metric have also been tested. You still need to run
actual training on real data — see steps below.

## 1. Setup (do this BEFORE the hackathon clock starts)

```bash
pip install -r requirements.txt
```

## 2. Get the Dataset (do this BEFORE the hackathon clock starts)

Download the hand-labeled Sen1Floods11 subset. Easiest option — Kaggle:
https://www.kaggle.com/datasets/smabrarrajin/sen1floods11-essentials

Unzip it so you have a folder structure like:
```
data/
  S1Hand/        <- SAR images (2-band GeoTIFF: VV, VH)
  LabelHand/     <- ground truth flood masks
```

If your download has different folder names, rename them to match, or edit
the `sar_dir` / `label_dir` paths at the top of `dataset.py`.

## 3. Train the Model

Quick smoke-test run first (confirms the pipeline works, ~5-10 min on CPU
Quick smoke-test run first (confirms the pipeline works end-to-end):
```bash
python train.py --data_dir "archive (1)/v1.2" --epochs 1 --max_samples 8 --img_size 64
```

Full split-based training run:
```bash
python train.py --data_dir "archive (1)/v1.2" \
  --train_split "archive (1)/v1.2/splits/flood_handlabeled/flood_train_data.csv" \
  --val_split "archive (1)/v1.2/splits/flood_handlabeled/flood_val_data.csv" \
  --epochs 5 --img_size 128 --batch_size 8
```

This saves the best model checkpoint to `models/flood_unet.pth`.

**Time-saving tip:** if you're short on time, fewer epochs (5-8) with the
pretrained ResNet34 encoder will still give reasonable results — you don't
need to train to full convergence for a hackathon demo.

## 4. Run Inference on a New Image (e.g. Tamil Nadu)

```bash
python infer.py --image path/to/tamilnadu_sar_image.tif --model models/flood_unet.pth
```

This prints the estimated flooded area in km².
This prints the estimated flooded area in km² and ranks the highest-priority
image tiles using predicted flood coverage multiplied by model confidence.

To get a real Tamil Nadu Sentinel-1 image, pull one from Google Earth Engine
or the Copernicus Data Space Ecosystem (see earlier chat notes) and export
it as a 2-band GeoTIFF (VV, VH) before running inference.

## 5. Launch the Website

```bash
streamlit run app.py
```

Open the local URL Streamlit gives you, upload a SAR GeoTIFF, and view the
predicted flood overlay, area/severity stats, and a ranked list of high-priority
flood zones.

## What's Included (MVP Scope)
- [x] Flood detection (U-Net segmentation)
- [x] Flooded area calculation (km²)
- [x] Basic severity classification (High/Medium/Low, by % area flooded)
- [x] Web dashboard with map overlay

## What's NOT Included Yet (documented as future work in the pitch)
- [ ] Population/infrastructure-weighted risk scoring
- [ ] Multi-zone ranked priority list for rescue teams
- [ ] Automated satellite data ingestion / alerting
- [ ] Before/after change detection (requires a separate pre-flood image,
      pulled manually via Earth Engine for the demo)

## Troubleshooting
- **CUDA not available / slow training:** the scripts auto-detect and fall
  back to CPU. Reduce `--img_size` and `--max_samples` for faster CPU runs.
- **GeoTIFF has more/fewer than 2 bands:** Sentinel-1 SAR chips should be
  VV + VH (2 bands). If your source image has extra bands, select just the
  VV/VH bands before running inference.
- **Model file not found in app.py:** make sure training finished and
  `models/flood_unet.pth` exists before launching Streamlit.
