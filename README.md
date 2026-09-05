# AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction Using Drone Imagery

A rapid technical prototype developed for the **Smart India Hackathon (SIH)** problem statement: *"AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction Using Drone Imagery"*.

---

## 🎯 Problem Statement & Objective

Manual cadastral surveying of urban land parcels is labor-intensive, time-consuming, and prone to boundary disputes. This prototype demonstrates an automated, AI-assisted pipeline to extract urban parcel boundaries from high-resolution drone imagery, generate regularized parcel polygons, compare them against historical cadastral survey maps, and flag potential encroachments or unauthorized land-use changes for surveyor verification.

> **Key Technical Distinction**: The system distinguishes between **Building Footprints** and **Legal Land Parcels**, employing contour regularization, boundary wall/fence detection, and historical cadastral alignment rather than treating building boundaries as property lines.

> **Prototype Disclaimer**: Pretrained AI segmentation models and geometric vectorization routines extract **AI-Assisted Candidate Parcel Boundaries**. They **do not independently determine legal land ownership or official cadastral boundaries**, but rather provide candidate geometric features for downstream GIS processing and certified surveyor verification. Area and perimeter measurements are strictly in **pixel units** (px², px) unless real-world georeferencing/GSD calibration is provided.

---

## 🔄 End-to-End Workflow

```
Aerial / Drone Imagery
        ↓
Image Preprocessing (Contrast, Tiling, RGB Normalization, Tensor Prep)  [COMPLETED]
        ↓
AI Feature & Semantic Segmentation (SegFormer Transformer ADE20K)       [COMPLETED]
        ↓
Boundary Extraction & Morphological Noise Filtering                    [COMPLETED]
        ↓
Parcel Polygon Vectorization, Regularization & GeoJSON Generation      [COMPLETED]
        ↓
GIS Geospatial Visualization (Folium / Web Map)
        ↓
Historical Cadastral Comparison & Change Detection
        ↓
Encroachment & Discrepancy Alerts
        ↓
Surveyor Verification & Quality Control
        ↓
Standard GeoJSON / Shapefile Export
```

---

## 📐 Geometric Boundary Extraction & Polygon Regularization

1. **Semantic Feature Filtering**: Extracts target structure semantic classes (`building`, `wall`, `fence`, `roof`).
2. **Morphological Cleanup**: Morphological opening (noise suppression) and closing (gap bridging) via OpenCV.
3. **Contour Extraction & Area Filtering**: Identifies external closed contours, filtering out speckles below configurable minimum pixel area thresholds.
4. **Polygon Regularization**: Douglas-Peucker geometric simplification (`preserve_topology=True`) converts jagged raster pixels into crisp vector edges.
5. **Topology Validation & Auto-Repair**: Shapely `is_valid` / `make_valid` checks ensure non-self-intersecting, topologically valid planar polygons.
6. **Sequential Parcel Metadata**: Assigns IDs (`P-001`, `P-002`...), computes pixel area (`px²`), pixel perimeter (`px`), vertex count, and solidity scores.
7. **GeoJSON Serialization**: Formats features into standard OGC-compliant GeoJSON `FeatureCollection` structures.

---

## 🛠️ Technology Stack

- **Frontend / UI**: Streamlit, Streamlit-Folium
- **Computer Vision & AI**: PyTorch, Hugging Face Transformers, OpenCV (`opencv-python-headless`), NumPy, Pillow
- **Geospatial & Vector**: Shapely, GeoPandas, Folium, GeoJSON
- **Data & Analytics**: Pandas

---

## 📁 Project Structure

```
sih-project/
│
├── app.py                             # Streamlit application entry point
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore specifications
│
├── src/                               # Core algorithmic modules
│   ├── __init__.py
│   ├── segmentation/                  # AI feature & semantic segmentation
│   │   ├── __init__.py
│   │   └── inference.py               # Model loading, segmentation, statistics, overlay
│   ├── geometry/                      # Boundary extraction & geometric processing
│   │   ├── __init__.py
│   │   └── boundary.py                # Contours, morphology, boundary overlays
│   ├── vectorization/                 # Polygon generation & regularization
│   │   ├── __init__.py
│   │   └── polygons.py                # Shapely validation, simplification, GeoJSON
│   ├── change_detection/              # Historical comparison & encroachment flagging
│   │   ├── __init__.py
│   │   └── compare.py
│   └── utils/                         # Geospatial & image processing utilities
│       ├── __init__.py
│       ├── helpers.py                 # GeoJSON export and formatters
│       └── image_processing.py        # Safe image loader, metadata, and AI preprocessor
│
├── data/
│   ├── input/                         # Raw input drone/aerial images
│   ├── cadastral/                     # Historical cadastral survey GeoJSON/records
│   └── demo/                          # Demo benchmark sample datasets
│
├── models/                            # Model weights & configurations
├── outputs/                           # Exported GeoJSON, maps, and reports
└── tests/                             # Automated smoke & unit tests
    ├── __init__.py
    ├── test_smoke.py
    ├── test_image_processing.py
    ├── test_segmentation.py
    └── test_boundary_and_polygons.py
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/vipingsit-hub/sih-project.git
cd sih-project
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```bash
python -m unittest discover tests
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## 📋 Development Roadmap

- [x] **Milestone 1**: Project architecture, environment setup, modular structure, smoke test suite.
- [x] **Milestone 2**: Aerial image input validation, metadata extraction, RGB normalization, and AI tensor preprocessing pipeline.
- [x] **Milestone 3**: AI-based feature segmentation using SegFormer Transformer, class statistics, and alpha-blended diagnostic overlays.
- [x] **Milestone 4**: Boundary extraction, morphological noise filtering, Douglas-Peucker polygon regularization, candidate parcel attributes, and GeoJSON export.
- [ ] **Milestone 5**: Interactive GIS visualization with Folium.
- [ ] **Milestone 6**: Historical cadastral comparison & encroachment detection.
- [ ] **Milestone 7**: Surveyor verification interface & GeoJSON export.
- [ ] **Milestone 8**: Final testing & demo dataset integration.

---

## 📄 License & Disclaimer

This prototype is built strictly as a technical proof-of-concept for the Smart India Hackathon. It is designed to serve as an **AI-assisted tool for certified surveyors** rather than independently generating legally binding cadastral land titles.
