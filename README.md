# AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction Using Drone Imagery

A rapid technical prototype developed for the **Smart India Hackathon (SIH)** problem statement: *"AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction Using Drone Imagery"*.

---

## 🎯 Problem Statement & Objective

Manual cadastral surveying of urban land parcels is labor-intensive, time-consuming, and prone to boundary disputes. This prototype demonstrates an automated, AI-assisted pipeline to extract urban parcel boundaries from high-resolution drone imagery, generate regularized candidate parcel polygons, perform spatial GIS comparison against existing/historical cadastral survey maps, and flag potential boundary discrepancies and encroachments for surveyor verification.

> **Key Technical Distinction**: The system distinguishes between **Building Footprints** and **Legal Land Parcels**, employing contour regularization, boundary wall/fence detection, and historical cadastral alignment rather than treating building boundaries as property lines.

> **Prototype Disclaimer & Legal Notice**: 
> The system identifies potential spatial discrepancies between existing cadastral reference geometry and AI-derived candidate geometry. It **does not establish legal property boundaries or confirm encroachment**. Final determination requires qualified surveyor and/or authorized authority verification. Area and perimeter measurements are in **pixel units** (px², px) unless real-world georeferencing/GSD calibration is provided.

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
Change & Potential Encroachment Detection (Extension Geometry)         [COMPLETED]
        ↓
Prototype Risk Classification (LOW / MEDIUM / HIGH)                    [COMPLETED]
        ↓
Surveyor Verification Queue & Human-in-the-Loop Audit                  [COMPLETED]
        ↓
Standard GeoJSON / Shapefile Export
```

---

## 🔍 Change Detection, Encroachment Analysis & Risk Classification

The change detection engine (`src/change_detection/compare.py`) analyzes the geometric relationship between reference cadastral polygons and AI-extracted candidate parcels:

1. **Spatial Change Metrics**:
   - **Intersection over Union (IoU)**: $\text{IoU} = \frac{\text{Area}(\text{Intersection})}{\text{Area}(\text{Union})}$, quantifying overall boundary overlap.
   - **Potential Extension Area**: Computes candidate structural footprint protruding outside the reference deed boundary:
     $$\text{Extension Geometry} = \text{Candidate Polygon} \setminus \text{Cadastral Reference}$$
   - **Missing Cadastral Area**: Identifies legal parcel portions without detected structures ($\text{Cadastral} \setminus \text{Candidate}$).
   - **Area Difference & Variance Percentage**: $\Delta \text{Area} = \text{Area}_{\text{cand}} - \text{Area}_{\text{cad}}$, and $\% \Delta = \frac{|\Delta \text{Area}|}{\text{Area}_{\text{cad}}} \times 100$.

2. **Discrepancy & Risk Classification Rules**:
   - `MATCH` (IoU $\ge$ 0.85): **LOW RISK** — Close geometric alignment.
   - `MINOR BOUNDARY MISMATCH` (0.60 $\le$ IoU < 0.85): **MEDIUM RISK** — Slight corner or edge variance.
   - `SIGNIFICANT BOUNDARY MISMATCH` (IoU < 0.60): **HIGH RISK** — Major footprint divergence.
   - `POTENTIAL ENCROACHMENT` (Extension Area $\ge$ 10% of Cadastral Area): **HIGH RISK** — Physical structure extends significantly beyond deed boundary.
   - `UNMATCHED` (No detected candidate): **MEDIUM RISK** — Vacant lot or undetected parcel.

3. **Human-in-the-Loop Surveyor Verification Queue**:
   - Prioritizes parcels by risk tier (HIGH $\rightarrow$ MEDIUM $\rightarrow$ LOW).
   - Allows certified land surveyors to inspect focused, zoomed-in overlays showing legal boundaries, current structures, and highlighted protrusion regions.
   - Provides interactive sign-off actions (`REVIEWED & VERIFIED`, `FLAGGED FOR ON-SITE SURVEY`, `RESET`) and field note tracking.

---

## 🗺️ GIS Visualization & Cadastral Comparison

The system integrates a lightweight, layered 2D GIS visualizer:

1. **Synthetic Demo Cadastral Dataset (`data/cadastral/demo_cadastral.geojson`)**:
   - Deterministic GeoJSON benchmark dataset created strictly for prototype evaluation.
   - Includes benchmark test cases: exact matches, minor boundary differences, substantial boundary mismatches, and potential lateral encroachments.
2. **Multi-Layer GIS Map Visualizer (`src/visualization/map.py`)**:
   - **Layer 1**: Existing Cadastral Parcels (Deep Blue outlines with soft fill)
   - **Layer 2**: AI Candidate Parcels (Bright Cyan dashed outlines)
   - **Layer 3**: Potential Encroachment / Extension Protrusions (Crimson red overlay)
   - **Parcel Detail Zoom**: Single-parcel deep-dive with bounding-box cropping and discrepancy annotations.

---

## 🎛️ Executive Dashboard & One-Click Demo Mode

The final dashboard (`app.py`) provides an executive-ready spatial decision support interface:

1. **⚡ One-Click AI Analysis Mode**:
   - Executes the complete 7-stage pipeline in a single unified run with live progress tracking.
   - Preprocesses input, runs SegFormer AI segmentation, extracts regularized boundaries, ingests cadastral survey deeds, executes spatial topology matching, isolates potential lateral protrusions, and generates the prioritized surveyor review queue.
2. **📁 Deterministic Demo Benchmark**:
   - One-click loading of benchmark aerial survey image (`data/demo/sample_drone_aerial.png`) and cadastral survey records (`data/cadastral/demo_cadastral.geojson`).
   - Generates deterministic benchmark results:
     - `P-001`: `MATCH` (IoU: 90.9%, Low Risk).
     - `P-002`: `POTENTIAL ENCROACHMENT` (IoU: 74.0%, Protrusion: 4,500 px², High Risk).
     - `P-003`: `POTENTIAL ENCROACHMENT` (IoU: 42.9%, Protrusion: 7,500 px², High Risk).
     - `P-004`: `SIGNIFICANT BOUNDARY MISMATCH` (IoU: 44.9%, Area Δ: -50%, High Risk).
     - `P-005`: `UNMATCHED` (No detected candidate structure, Medium Risk).
3. **📊 Executive KPI Section**:
   - Real-time metric cards: Total Cadastral Parcels, AI Candidates, Verified Matches, Discrepancies, High-Risk Flags, and Pending Surveyor Queue.
4. **🗺️ Main GIS Map & Priority Findings**:
   - Central multi-layer visualizer (Blue: Cadastral Deeds, Cyan: AI Candidates, Red: Potential Encroachment Protrusions).
   - Priority Findings table sorted by risk severity with interactive plot highlighting.
5. **🔍 Surveyor Inspection & Sign-Off Station**:
   - High-resolution cropped zoom overlay comparing deed lines against physical structures and highlighted protrusion regions.
   - Human-in-the-loop action buttons (`FLAG FOR ON-SITE SURVEY`, `MARK AS REVIEWED`, `RESET STATUS`) and field note tracking with session state persistence.
6. **⚙️ AI Pipeline Diagnostics**:
   - Step-by-step visualizer for technical evaluation (Raw Aerial $\rightarrow$ SegFormer ADE20K Semantics $\rightarrow$ Contours & Morphology $\rightarrow$ Douglas-Peucker Vectors).
7. **📥 Reports & Multi-Format Export**:
   - Candidate Parcels GeoJSON (`candidate_parcels.geojson`).
   - Flagged Encroachments GeoJSON (`potential_encroachments.geojson`).
   - Surveyor Audit Log CSV (`cadastral_surveyor_audit_report.csv`).

---

## 🌐 Coordinate Reference System & Georeferencing Limitation

- **Current Prototype Coordinates**: Standardized local/demo pixel coordinate space.
- **Georeferencing Roadmap**: Designed with a clean modular abstraction (`src/geospatial/`, `src/change_detection/`) that enables replacing local coordinates with real-world georeferenced GeoTIFF imagery (EPSG:4326 / UTM projected coordinates) and official municipal GIS layers.

---

## 🛠️ Technology Stack

- **Frontend / UI**: Streamlit
- **Computer Vision & AI**: PyTorch, Hugging Face Transformers, OpenCV (`opencv-python-headless`), NumPy, Pillow
- **Geospatial & Vector Processing**: Shapely, GeoJSON
- **Data & Analytics**: Pandas

---

## 📁 Project Structure

```
sih-project/
│
├── app.py                             # Streamlit Executive Dashboard (One-Click Demo & Surveyor Station)
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
│   ├── change_detection/              # Change detection & risk classification
│   │   ├── __init__.py
│   │   └── compare.py                 # Extension area geometry, risk classification, surveyor queue
│   ├── visualization/                 # GIS mapping & discrepancy rendering
│   │   ├── __init__.py
│   │   └── map.py                     # Multi-layer GIS map & parcel detail deep-dive rendering
│   └── utils/                         # Geospatial & image processing utilities
│       ├── __init__.py
│       ├── helpers.py                 # GeoJSON export, encroachment serializer, CSV formatters
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
└── tests/                             # Automated smoke & unit tests (37 tests)
    ├── __init__.py
    ├── test_smoke.py                  # End-to-end pipeline integration smoke test
    ├── test_image_processing.py       # Image input, metadata & tensor tests
    ├── test_segmentation.py           # SegFormer AI model inference tests
    ├── test_boundary_and_polygons.py  # Contour morphology & Shapely vector tests
    ├── test_cadastral_and_comparison.py # GeoJSON validation & IoU tests
    └── test_change_detection.py       # Change metrics, risk tiers & surveyor queue tests
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
- [x] **Milestone 5**: Layered GIS visualization, synthetic cadastral ingestion, spatial overlap (IoU) comparison.
- [x] **Milestone 6**: Temporal change detection, potential encroachment analysis, risk tiers, and surveyor verification queue.
- [x] **Milestone 7**: Final hackathon dashboard, one-click demo mode, surveyor inspection station, and multi-format report exports.

---

## 📄 License & Disclaimer

This prototype is built strictly as a technical proof-of-concept for the Smart India Hackathon. It is designed to serve as an **AI-assisted tool for certified surveyors** rather than independently generating legally binding cadastral land titles.
