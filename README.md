# AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction Using Drone Imagery

A rapid technical prototype developed for the **Smart India Hackathon (SIH)** problem statement: *"AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction Using Drone Imagery"*.

---

## 🎯 Problem Statement & Objective

Manual cadastral surveying of urban land parcels is labor-intensive, time-consuming, and prone to boundary disputes. This prototype demonstrates an automated, AI-assisted pipeline to extract urban parcel boundaries from high-resolution drone imagery, generate regularized candidate parcel polygons, perform spatial GIS comparison against existing/historical cadastral survey maps, and flag potential boundary discrepancies and encroachments for surveyor verification.

> **Key Technical Distinction**: The system distinguishes between **Building Footprints** and **Legal Land Parcels**, employing contour regularization, boundary wall/fence detection, and historical cadastral alignment rather than treating building boundaries as property lines.

> **Prototype Disclaimer & Legal Notice**: 
> The prototype identifies potential spatial discrepancies using AI-derived candidate geometry and existing cadastral reference geometry. It **does not establish legal property boundaries or confirm legal encroachment**. All outputs represent **candidate geometry** and **potential discrepancies** intended to assist certified land surveyors and GIS professionals. Area and perimeter measurements are in **pixel units** (px², px) unless real-world georeferencing/GSD calibration is provided.

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
GIS Geospatial Visualization & Demo Cadastral Ingestion                [COMPLETED]
        ↓
Spatial Topology Comparison & Geometric Similarity (IoU)               [COMPLETED]
        ↓
Discrepancy Categorization & Potential Encroachment Alerts             [COMPLETED]
        ↓
Surveyor Verification & Quality Control
        ↓
Standard GeoJSON / Shapefile Export
```

---

## 🗺️ GIS Visualization & Cadastral Comparison

The system integrates a lightweight, layered 2D GIS visualizer and spatial topology comparison engine:

1. **Synthetic Demo Cadastral Dataset (`data/cadastral/demo_cadastral.geojson`)**:
   - Deterministic GeoJSON benchmark dataset created strictly for prototype evaluation.
   - Includes benchmark test cases: exact matches, minor boundary differences, substantial boundary mismatches, and potential lateral encroachments.
2. **Spatial Topology Analysis (`src/geospatial/comparison.py`)**:
   - **Intersection over Union (IoU)**: Evaluates geometric overlap similarity ($IoU = \frac{\text{Area}(\text{Intersection})}{\text{Area}(\text{Union})}$).
   - **Area Difference & Excess Protrusion**: Calculates candidate geometry protrusion extending beyond reference cadastral boundaries.
   - **Discrepancy Categorization**:
     - `MATCH` (IoU $\ge$ 0.85)
     - `MINOR MISMATCH` (0.60 $\le$ IoU < 0.85)
     - `BOUNDARY MISMATCH` (IoU < 0.60)
     - `POTENTIAL ENCROACHMENT` (Candidate extends beyond cadastral parcel by $\ge$ 10% excess area)
     - `UNMATCHED` (No corresponding cadastral or AI candidate geometry)
3. **Multi-Layer GIS Map Visualizer (`src/visualization/map.py`)**:
   - **Layer 1**: Existing Cadastral Parcels (Deep Blue outlines with soft fill)
   - **Layer 2**: AI Candidate Parcels (Bright Cyan dashed outlines)
   - **Layer 3**: Potential Encroachment / Discrepancy Zones (Crimson red overlay)
   - Interactive parcel selection, summary metrics, and side-by-side comparison tables.

---

## 🌐 Coordinate Reference System & Georeferencing Limitation

- **Current Prototype Coordinates**: Evaluated in standardized local/demo pixel coordinate space.
- **Georeferencing Roadmap**: Designed with a clean modular abstraction (`src/geospatial/`) that directly enables replacing local coordinates with real-world georeferenced GeoTIFF imagery (EPSG:4326 / UTM projected coordinates) and official state GIS cadastral layers in future deployment phases.

---

## 🛠️ Technology Stack

- **Frontend / UI**: Streamlit
- **Computer Vision & AI**: PyTorch, Hugging Face Transformers, OpenCV (`opencv-python-headless`), NumPy, Pillow
- **Geospatial & Vector Processing**: Shapely, Matplotlib / Folium, GeoJSON
- **Data & Analytics**: Pandas

---

## 📁 Project Structure

```
sih-project/
│
├── app.py                             # Streamlit application entry point (End-to-End Workflow)
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore specifications
│
├── src/                               # Core algorithmic modules
│   ├── __init__.py
│   ├── segmentation/                  # AI feature & semantic segmentation
│   │   ├── __init__.py
│   │   └── inference.py               # SegFormer inference, class stats, alpha overlays
│   ├── geometry/                      # Boundary extraction & geometric processing
│   │   ├── __init__.py
│   │   └── boundary.py                # Contours, morphology, boundary overlays
│   ├── vectorization/                 # Polygon generation & regularization
│   │   ├── __init__.py
│   │   └── polygons.py                # Shapely validation, Douglas-Peucker simplification, GeoJSON
│   ├── geospatial/                    # Cadastral loading & spatial topology comparison
│   │   ├── __init__.py
│   │   ├── cadastral.py               # Safe GeoJSON parser, geometry validator & stats
│   │   └── comparison.py              # IoU, spatial overlap, discrepancy & encroachment classifier
│   ├── visualization/                 # GIS mapping & discrepancy rendering
│   │   ├── __init__.py
│   │   └── map.py                     # Multi-layer GIS map rendering & legend generator
│   └── utils/                         # Geospatial & image processing utilities
│       ├── __init__.py
│       ├── helpers.py                 # GeoJSON export and formatters
│       └── image_processing.py        # Safe image loader, metadata, and AI preprocessor
│
├── data/
│   ├── input/                         # Raw input drone/aerial images
│   ├── cadastral/                     # Synthetic demo cadastral dataset (GeoJSON)
│   │   └── demo_cadastral.geojson     # Benchmark parcels (P-001 to P-005)
│   └── demo/                          # Demo benchmark sample datasets
│
├── models/                            # Model weights & configurations
├── outputs/                           # Exported GeoJSON, maps, and reports
└── tests/                             # Automated smoke & unit tests (31 tests)
    ├── __init__.py
    ├── test_smoke.py                  # End-to-end pipeline integration smoke test
    ├── test_image_processing.py       # Image input, metadata & tensor tests
    ├── test_segmentation.py           # SegFormer AI model inference tests
    ├── test_boundary_and_polygons.py  # Contour morphology & Shapely vector tests
    └── test_cadastral_and_comparison.py # GeoJSON validation, IoU & encroachment tests
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
- [x] **Milestone 5**: Layered GIS visualization, synthetic cadastral ingestion, spatial overlap (IoU) comparison, and potential encroachment classification.
- [ ] **Milestone 6**: Temporal change detection & historical survey alignment.
- [ ] **Milestone 7**: Interactive surveyor verification tool, manual boundary vertex adjustment & official export.

---

## 📄 License & Disclaimer

This prototype is built strictly as a technical proof-of-concept for the Smart India Hackathon. It is designed to serve as an **AI-assisted tool for certified surveyors** rather than independently generating legally binding cadastral land titles.
