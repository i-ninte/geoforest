"""
Ghana Forest Dataset Builder
=============================
Downloads Sentinel-2 RGB+NIR (B4, B3, B2, B8) for ALL of Ghana,
tiles into 512×512 patches, generates PNG previews + NDVI labels,
and packages everything ready for Kaggle upload.

Workflow:
  Step 1 — Submit GEE export tasks (grid of ~80 cells across Ghana)
  Step 2 — Wait for Drive exports (2-5 days for full Ghana)
  Step 3 — Process downloaded GeoTIFFs → patches + PNGs + labels
  Step 4 — Upload to Kaggle

Usage:
  python kaggle_dataset_builder.py submit          # Step 1
  python kaggle_dataset_builder.py status          # Check task progress
  python kaggle_dataset_builder.py process <dir>   # Step 3 (after Drive sync)
  python kaggle_dataset_builder.py upload          # Step 4
  python kaggle_dataset_builder.py all             # Steps 1+3+4 (automated)

Requirements:
  pip install earthengine-api geemap rasterio numpy pillow tqdm kaggle
"""

import ee
import os
import json
import time
import math
import hashlib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

# !! EDIT THESE !!
KAGGLE_USERNAME = "your_kaggle_username"      # your Kaggle username
DATASET_TITLE   = "Ghana Sentinel-2 Forest Dataset"
DATASET_SLUG    = "ghana-sentinel2-forest"    #

# Composite period — one cloud-free median over this range
# Longer range = cleaner composites (fewer cloud holes)
COMPOSITE_START = "2023-01-01"
COMPOSITE_END   = "2024-01-01"

# Output structure
BASE_DIR        = Path("./ghana_s2_dataset")
GEOTIFF_DIR     = BASE_DIR / "geotiffs"       # raw GeoTIFFs from Drive
PATCHES_DIR     = BASE_DIR / "patches"        # 512×512 GeoTIFF patches
PREVIEWS_DIR    = BASE_DIR / "previews"       # PNG previews (RGB)
LABELS_FILE     = BASE_DIR / "ndvi_labels.csv"
METADATA_FILE   = BASE_DIR / "metadata.json"
TASKS_FILE      = BASE_DIR / "gee_tasks.json"

# Tile settings
PATCH_SIZE      = 512    # pixels
SCALE           = 10     # metres/pixel (Sentinel-2 native)
MAX_CLOUD       = 15     # max cloud cover % per scene (strict = cleaner)
NODATA_SKIP     = 0.40   # skip patch if >40% pixels are masked

# NDVI threshold for "forest / dense vegetation" label
NDVI_FOREST     = 0.40
NDVI_WOODLAND   = 0.25

# Ghana bounding box — we grid this into cells for GEE
# (lon_min, lat_min, lon_max, lat_max)
GHANA_BBOX = (-3.25, 4.74, 1.20, 11.17)

# Grid resolution: ~0.5° cells ≈ ~55km × 55km each
# Ghana is ~4.5° wide × 6.4° tall → ~9 cols × 13 rows = ~117 cells
# (coastal cells will be mostly ocean — GEE skips empty ones automatically)
GRID_STEP = 0.5

# Bands: B4=Red, B3=Green, B2=Blue, B8=NIR
BANDS = ["B4", "B3", "B2", "B8"]

# ── DIRECTORY SETUP ───────────────────────────────────────────────────────────

def setup_dirs():
    for d in [BASE_DIR, GEOTIFF_DIR, PATCHES_DIR, PREVIEWS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ── GEE HELPERS ───────────────────────────────────────────────────────────────

def init_gee():
    try:
        ee.Initialize()
        print("✓ GEE authenticated")
    except Exception:
        print("Opening browser for GEE authentication…")
        ee.Authenticate()
        ee.Initialize()
        print("✓ GEE authenticated")


def mask_clouds(image):
    """Mask clouds + cirrus using QA60 bitmask, then scale to [0,1]."""
    qa = image.select("QA60")
    mask = (
        qa.bitwiseAnd(1 << 10).eq(0)   # opaque clouds
        .And(qa.bitwiseAnd(1 << 11).eq(0))  # cirrus
    )
    return image.updateMask(mask).divide(10000)


def build_composite(geom):
    """
    Build a cloud-free Sentinel-2 SR median composite.
    Uses harmonised collection (consistent across sensor generations).
    """
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(COMPOSITE_START, COMPOSITE_END)
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
        .map(mask_clouds)
        .select(BANDS)
        .median()
        .clip(geom)
    )
    return col


def make_ghana_grid():
    """
    Generate a list of (col, row, bbox) grid cells covering Ghana.
    Each cell is GRID_STEP × GRID_STEP degrees.
    """
    lon_min, lat_min, lon_max, lat_max = GHANA_BBOX
    cells = []
    col = 0
    lon = lon_min
    while lon < lon_max:
        row = 0
        lat = lat_min
        while lat < lat_max:
            cells.append({
                "col":     col,
                "row":     row,
                "lon_min": round(lon, 4),
                "lat_min": round(lat, 4),
                "lon_max": round(min(lon + GRID_STEP, lon_max), 4),
                "lat_max": round(min(lat + GRID_STEP, lat_max), 4),
                "id":      f"c{col:02d}r{row:02d}",
            })
            lat += GRID_STEP
            row += 1
        lon += GRID_STEP
        col += 1
    return cells


# ── STEP 1: SUBMIT GEE TASKS ──────────────────────────────────────────────────

def submit_export_tasks():
    """
    Submit one GEE export task per grid cell.
    Exports land in Google Drive → GhanaS2Dataset/
    """
    setup_dirs()
    init_gee()

    cells = make_ghana_grid()
    print(f"\nGhana grid: {len(cells)} cells ({GRID_STEP}° × {GRID_STEP}°)")
    print(f"Composite period: {COMPOSITE_START} → {COMPOSITE_END}")
    print(f"Drive folder: GhanaS2Dataset\n")

    # Load previously submitted tasks to avoid duplicates
    existing = {}
    if TASKS_FILE.exists():
        existing = {t["cell_id"]: t for t in json.loads(TASKS_FILE.read_text())}

    tasks = list(existing.values())
    submitted = 0

    for cell in tqdm(cells, desc="Submitting tasks"):
        cid = cell["id"]
        if cid in existing:
            continue   # already submitted

        geom  = ee.Geometry.Rectangle(
            [cell["lon_min"], cell["lat_min"], cell["lon_max"], cell["lat_max"]]
        )
        image = build_composite(geom)
        label = f"ghana_s2_{cid}_{COMPOSITE_START[:4]}"

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=label,
            folder="GhanaS2Dataset",
            fileNamePrefix=label,
            region=geom,
            scale=SCALE,
            crs="EPSG:4326",
            maxPixels=1e10,
            fileFormat="GeoTIFF",
        )
        task.start()

        tasks.append({
            "cell_id":     cid,
            "task_id":     task.id,
            "description": label,
            "cell":        cell,
            "submitted_at": datetime.utcnow().isoformat(),
            "status":      "SUBMITTED",
        })
        submitted += 1

    TASKS_FILE.write_text(json.dumps(tasks, indent=2))
    print(f"\n✓ {submitted} new tasks submitted ({len(tasks)} total)")
    print(f"  Task log: {TASKS_FILE}")
    print(f"\nMonitor at: https://code.earthengine.google.com/tasks")
    print(f"Run 'python {__file__} status' to check progress.")


# ── STEP 1b: CHECK TASK STATUS ────────────────────────────────────────────────

def check_task_status():
    """Poll GEE for the status of all submitted tasks."""
    if not TASKS_FILE.exists():
        print("No tasks file found. Run 'submit' first.")
        return

    init_gee()
    tasks = json.loads(TASKS_FILE.read_text())

    counts = {"COMPLETED": 0, "RUNNING": 0, "READY": 0, "FAILED": 0, "CANCELLED": 0, "OTHER": 0}
    updated = []

    for t in tqdm(tasks, desc="Checking tasks"):
        try:
            status = ee.batch.Task(t["task_id"]).status()["state"]
        except Exception:
            status = t.get("status", "UNKNOWN")
        t["status"] = status
        counts[status if status in counts else "OTHER"] += 1
        updated.append(t)

    TASKS_FILE.write_text(json.dumps(updated, indent=2))

    total = len(tasks)
    done  = counts["COMPLETED"]
    print(f"\nTask status ({total} total):")
    for k, v in counts.items():
        if v > 0:
            bar = "█" * int(20 * v / total)
            print(f"  {k:<12} {v:>4}  {bar}")
    print(f"\nProgress: {done}/{total} ({100*done//max(total,1)}%)")

    if counts["FAILED"] > 0:
        failed = [t["description"] for t in tasks if t["status"] == "FAILED"]
        print(f"\nFailed tasks ({len(failed)}):")
        for f in failed[:10]:
            print(f"  · {f}")


# ── STEP 2: PROCESS DOWNLOADED GEOTIFFS ──────────────────────────────────────

def process_geotiffs(geotiff_dir: Path = None):
    """
    For each GeoTIFF in geotiff_dir:
      1. Slice into 512×512 patches
      2. Generate PNG preview (gamma-corrected RGB)
      3. Compute per-patch NDVI stats and label
    Then write ndvi_labels.csv and metadata.json.
    """
    import rasterio
    from PIL import Image as PILImage

    setup_dirs()
    src_dir = geotiff_dir or GEOTIFF_DIR
    tifs    = sorted(src_dir.glob("*.tif")) + sorted(src_dir.glob("*.TIF"))

    if not tifs:
        print(f"No GeoTIFF files found in {src_dir}")
        print("Sync your Google Drive GhanaS2Dataset/ folder here first.")
        return

    print(f"\nProcessing {len(tifs)} GeoTIFF(s) from {src_dir}")

    all_labels  = []
    total_patches = 0
    total_skipped = 0

    for tif_path in tqdm(tifs, desc="GeoTIFFs"):
        with rasterio.open(tif_path) as src:
            W, H  = src.width, src.height
            meta  = src.meta.copy()
            transform = src.transform

            n_cols = W // PATCH_SIZE
            n_rows = H // PATCH_SIZE

            for row_i in range(n_rows):
                for col_i in range(n_cols):
                    col_off = col_i * PATCH_SIZE
                    row_off = row_i * PATCH_SIZE

                    window = rasterio.windows.Window(col_off, row_off, PATCH_SIZE, PATCH_SIZE)
                    data   = src.read(window=window).astype(np.float32)

                    # Skip mostly-masked / nodata patches
                    nodata_frac = np.mean(data == 0)
                    if nodata_frac > NODATA_SKIP:
                        total_skipped += 1
                        continue

                    # ── Patch ID ──────────────────────────────────────────
                    stem     = tif_path.stem
                    patch_id = f"{stem}_r{row_i:04d}_c{col_i:04d}"

                    # ── Save GeoTIFF patch ─────────────────────────────────
                    win_transform = rasterio.windows.transform(window, transform)
                    patch_meta = meta.copy()
                    patch_meta.update({
                        "width":     PATCH_SIZE,
                        "height":    PATCH_SIZE,
                        "transform": win_transform,
                    })
                    patch_path = PATCHES_DIR / f"{patch_id}.tif"
                    with rasterio.open(patch_path, "w", **patch_meta) as dst:
                        dst.write(data.astype(meta["dtype"]))

                    # ── NDVI computation ──────────────────────────────────
                    # Band order: B4=idx0(Red), B3=idx1(Green), B2=idx2(Blue), B8=idx3(NIR)
                    red = data[0].copy()
                    nir = data[3].copy()
                    # Avoid division by zero
                    denom = nir + red
                    denom[denom == 0] = np.nan
                    ndvi = (nir - red) / denom
                    ndvi[data[0] == 0] = np.nan   # mask nodata

                    mean_ndvi = float(np.nanmean(ndvi))
                    std_ndvi  = float(np.nanstd(ndvi))
                    pct_forest    = float(np.nanmean(ndvi >= NDVI_FOREST))
                    pct_woodland  = float(np.nanmean((ndvi >= NDVI_WOODLAND) & (ndvi < NDVI_FOREST)))

                    # Dominant label
                    if pct_forest >= 0.30:
                        label = "dense_forest"
                    elif pct_forest + pct_woodland >= 0.30:
                        label = "woodland"
                    elif mean_ndvi >= 0.10:
                        label = "vegetation"
                    else:
                        label = "non_vegetation"

                    # ── PNG preview (gamma-corrected RGB) ─────────────────
                    rgb = data[:3][[0, 1, 2]]   # B4, B3, B2 → R, G, B
                    # Clip to [0, 0.3] (typical reflectance range) and scale to [0,255]
                    rgb = np.clip(rgb, 0, 0.3) / 0.3
                    # Gamma correction (makes forests visually richer)
                    rgb = np.power(rgb, 0.5)
                    rgb = (rgb * 255).astype(np.uint8)
                    img = PILImage.fromarray(rgb.transpose(1, 2, 0), mode="RGB")
                    preview_path = PREVIEWS_DIR / f"{patch_id}.png"
                    img.save(preview_path, optimize=True)

                    # ── Record label ──────────────────────────────────────
                    all_labels.append({
                        "patch_id":       patch_id,
                        "source_tif":     tif_path.name,
                        "row":            row_i,
                        "col":            col_i,
                        "mean_ndvi":      round(mean_ndvi, 4),
                        "std_ndvi":       round(std_ndvi, 4),
                        "pct_forest":     round(pct_forest, 4),
                        "pct_woodland":   round(pct_woodland, 4),
                        "label":          label,
                        "nodata_frac":    round(float(nodata_frac), 4),
                        "patch_tif":      f"patches/{patch_id}.tif",
                        "preview_png":    f"previews/{patch_id}.png",
                    })

                    total_patches += 1

    # ── Write labels CSV ──────────────────────────────────────────────────────
    import csv
    if all_labels:
        fieldnames = list(all_labels[0].keys())
        with open(LABELS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_labels)

    # ── Write metadata JSON ───────────────────────────────────────────────────
    label_counts = {}
    for row in all_labels:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1

    metadata = {
        "dataset_title":     DATASET_TITLE,
        "composite_start":   COMPOSITE_START,
        "composite_end":     COMPOSITE_END,
        "satellite":         "Sentinel-2 SR Harmonized",
        "bands":             {"1": "B4 (Red)", "2": "B3 (Green)", "3": "B2 (Blue)", "4": "B8 (NIR)"},
        "resolution_m":      SCALE,
        "patch_size_px":     PATCH_SIZE,
        "patch_size_km":     round(PATCH_SIZE * SCALE / 1000, 2),
        "total_patches":     total_patches,
        "skipped_patches":   total_skipped,
        "label_distribution": label_counts,
        "ndvi_thresholds":   {
            "dense_forest": f">= {NDVI_FOREST}  (≥30% of patch)",
            "woodland":     f">= {NDVI_WOODLAND} combined",
            "vegetation":   "mean NDVI >= 0.10",
            "non_vegetation": "< 0.10",
        },
        "coverage":          "All of Ghana",
        "created_at":        datetime.utcnow().isoformat(),
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"✓ Processing complete")
    print(f"  Patches saved:   {total_patches}")
    print(f"  Patches skipped: {total_skipped} (>{NODATA_SKIP*100:.0f}% nodata)")
    print(f"\n  Label distribution:")
    for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * v / max(total_patches, 1)
        bar = "█" * int(pct / 2)
        print(f"    {k:<20} {v:>6}  {pct:5.1f}%  {bar}")
    print(f"\n  Labels CSV:  {LABELS_FILE}")
    print(f"  Metadata:    {METADATA_FILE}")
    print(f"{'─'*50}")


# ── STEP 3: UPLOAD TO KAGGLE ──────────────────────────────────────────────────

def upload_to_kaggle():
    """
    Package the processed dataset and upload to Kaggle.
    Requires ~/.kaggle/kaggle.json with your API credentials.
    Get it from: https://www.kaggle.com/settings → API → Create New Token
    """
    # Write dataset-metadata.json for Kaggle API
    kaggle_meta = {
        "title":       DATASET_TITLE,
        "id":          f"{KAGGLE_USERNAME}/{DATASET_SLUG}",
        "licenses":    [{"name": "CC0-1.0"}],
        "description": (
            f"Sentinel-2 SR (B4/B3/B2/B8) median composite for all of Ghana "
            f"({COMPOSITE_START} to {COMPOSITE_END}). "
            f"512×512px patches at 10m resolution. "
            f"Includes PNG previews and auto-generated NDVI vegetation labels "
            f"(dense_forest / woodland / vegetation / non_vegetation)."
        ),
    }
    meta_path = BASE_DIR / "dataset-metadata.json"
    meta_path.write_text(json.dumps(kaggle_meta, indent=2))

    print(f"\nUploading to Kaggle as {KAGGLE_USERNAME}/{DATASET_SLUG}…")
    print("(This may take a while for a large dataset)\n")

    # Check if dataset exists already (update vs create)
    import subprocess
    result = subprocess.run(
        ["kaggle", "datasets", "status", f"{KAGGLE_USERNAME}/{DATASET_SLUG}"],
        capture_output=True, text=True
    )

    if "No such dataset" in result.stdout or result.returncode != 0:
        # Create new dataset
        cmd = ["kaggle", "datasets", "create", "-p", str(BASE_DIR), "--dir-mode", "zip"]
        action = "Creating"
    else:
        # Update existing dataset
        cmd = ["kaggle", "datasets", "version", "-p", str(BASE_DIR),
               "-m", f"Updated {datetime.utcnow().strftime('%Y-%m-%d')}",
               "--dir-mode", "zip"]
        action = "Updating"

    print(f"{action} dataset…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)
    else:
        print(f"\n✓ Dataset available at:")
        print(f"  https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ghana Sentinel-2 Kaggle Dataset Builder")
    parser.add_argument("command", choices=["submit", "status", "process", "upload", "all"],
                        help="Command to run")
    parser.add_argument("geotiff_dir", nargs="?", default=None,
                        help="Directory containing downloaded GeoTIFFs (for 'process')")
    args = parser.parse_args()

    if args.command == "submit":
        submit_export_tasks()

    elif args.command == "status":
        check_task_status()

    elif args.command == "process":
        src = Path(args.geotiff_dir) if args.geotiff_dir else GEOTIFF_DIR
        process_geotiffs(src)

    elif args.command == "upload":
        upload_to_kaggle()

    elif args.command == "all":
        submit_export_tasks()
        print("\nWaiting for tasks to complete…")
        print("(Re-run with 'status' to check, then 'process' once Drive is synced)")


if __name__ == "__main__":
    main()
