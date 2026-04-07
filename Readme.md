

# GeoForest 🌳

**Satellite-based monitoring of deforestation and illegal mining (galamsey) in Ghana using deep learning and multispectral satellite imagery**

GeoForest is a geospatial machine learning project that uses satellite imagery and computer vision to detect **forest degradation, illegal logging, and galamsey expansion** in Ghana.

The project combines **multispectral satellite data, vegetation indices, object detection models, and temporal analysis** to monitor forest canopy changes over time.

---

# Motivation

Illegal mining (galamsey) and unsustainable logging contribute significantly to **forest loss and environmental degradation in Ghana**. Monitoring these activities at scale is challenging using traditional ground-based methods.

Satellite imagery provides an opportunity to build **automated monitoring systems** capable of detecting environmental change across large geographic regions.

GeoForest aims to build an **AI-powered monitoring pipeline** that can:

* detect vegetation loss
* identify possible mining expansion
* track forest degradation over time
* support environmental monitoring efforts

The project leverages data from **Sentinel-2**, which provides multispectral imagery suitable for vegetation monitoring.

---

# Project Architecture

The system follows a geospatial ML pipeline:

```
Satellite imagery (Sentinel-2)
        ↓
Data preprocessing & tiling
        ↓
Vegetation index computation (NDVI)
        ↓
Object detection model (RF-DETR)
        ↓
Temporal change analysis
        ↓
Forest loss / galamsey detection
```

The system integrates **two primary signals**:

1. **Computer vision detection**
2. **Vegetation health monitoring**

Combining these improves detection reliability.

---

# Dataset

The dataset used for this project is publicly available on Kaggle.

**Dataset**

[https://www.kaggle.com/datasets/kwabenaobeng/ghana-sentinel2-forest](https://www.kaggle.com/datasets/kwabenaobeng/ghana-sentinel2-forest)

The dataset contains **multispectral Sentinel-2 image tiles covering regions of Ghana**, prepared for machine learning workflows.

---

# Dataset Structure

```
ghana-sentinel2-forest/

images/
    tile_0001.tif
    tile_0002.tif
    tile_0003.tif

labels/
    ndvi_stats.csv

metadata/
    tiles.geojson
    timestamps.csv
```

---

# Image Format

Each satellite tile contains:

* **GeoTIFF format**
* **512 × 512 pixels**
* **4 spectral bands**

Band configuration:

```
Band 1 → Blue
Band 2 → Green
Band 3 → Red
Band 4 → Near Infrared (NIR)
```

Using **RGB + NIR** allows vegetation monitoring and spectral analysis.

---

# Vegetation Monitoring

Vegetation health is estimated using the **Normalized Difference Vegetation Index (NDVI)**.

NDVI measures vegetation density and health by comparing red and near-infrared reflectance.

NDVI = \frac{NIR - Red}{NIR + Red}

Higher NDVI values generally indicate healthier vegetation.

The dataset includes **precomputed NDVI statistics for each tile**, enabling temporal vegetation monitoring.

Example:

```
tile_id,date,ndvi_mean,ndvi_std,vegetation_cover
0001,2025-01-15,0.82,0.03,0.91
```

---

# Model Approach

The project uses **RF-DETR**, a transformer-based object detection architecture.

The model will be trained to detect features such as:

* forest canopy clusters
* cleared land
* potential mining sites
* vegetation loss regions

Detection outputs will then be combined with vegetation trends for temporal monitoring.

Pipeline:

```
GeoTIFF image tiles
        ↓
Load multispectral bands
        ↓
RF-DETR inference
        ↓
feature detection
        ↓
temporal comparison
        ↓
change detection
```

---

# Dataset Generation

The dataset was generated using **Google Earth Engine**.

The dataset builder notebook:

```
notebooks/ghana-forest-dataset-builder.ipynb
```

performs the following steps:

1. Query Sentinel-2 imagery
2. Filter scenes by cloud cover
3. Generate satellite image tiles
4. Export GeoTIFF patches
5. Compute vegetation indices
6. Generate dataset metadata

---

# Research Objectives

GeoForest explores several research directions:

* satellite-based forest monitoring
* deep learning for environmental change detection
* multispectral object detection
* temporal analysis of vegetation change

The project investigates how **computer vision models can be applied to geospatial data** for environmental monitoring tasks.

---

# Future Work

Planned extensions include:

* training RF-DETR on canopy and land cover transitions
* building automated **deforestation detection models**
* identifying potential galamsey sites
* developing time-series change detection models
* integrating **near-real-time satellite monitoring pipelines**

Long term, the project aims to support **automated forest monitoring systems** for environmental analysis.

---

# Repository Structure

```
geoforest/

datasets/
notebooks/
models/
training/
inference/
utils/
README.md
```

---

# Reproducibility

To reproduce the dataset and experiments:

1. Download the dataset from Kaggle.
2. Install project dependencies.
3. Run the dataset builder notebook if regenerating data.
4. Train the detection model using the training scripts.

Future releases will include full training pipelines and evaluation benchmarks.

---

# License

This repository is released for **research and educational purposes**.

