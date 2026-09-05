# AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction Using Drone Imagery

A rapid technical prototype developed for the **Smart India Hackathon (SIH)** problem statement: *"AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction Using Drone Imagery"*.

---

## 🎯 Problem Statement & Objective

Manual cadastral surveying of urban land parcels is labor-intensive, time-consuming, and prone to boundary disputes. This prototype demonstrates an automated, AI-assisted pipeline to extract urban parcel boundaries from high-resolution drone imagery, generate regularized parcel polygons, compare them against historical cadastral survey maps, and flag potential encroachments or unauthorized land-use changes for surveyor verification.

> **Key Distinction**: The system distinguishes between **Building Footprints** and **Legal Land Parcels**, employing contour regularization, boundary wall/fence detection, and historical cadastral alignment rather than treating building boundaries as property lines.

---

## 🔄 End-to-End Workflow

```
Aerial / Drone Imagery
        ↓
Image Preprocessing (Contrast, Tiling, Normalization)
        ↓
AI Feature & Boundary Segmentation (Deep Learning / CV)
        ↓
Boundary Extraction & Regularization
        ↓
Parcel Polygon Vectorization & Topology Validation
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

## 🛠️ Technology Stack

- **Frontend / UI**: Streamlit, Streamlit-Folium
- **Geospatial & Vector**: GeoPandas, Shapely, Folium, GeoJSON
- **Computer Vision & AI**: PyTorch, OpenCV, NumPy, Pillow
- **Data & Analytics**: Pandas

---

## 📁 Project Structure

```
sih-project/
│
├── app.py                      # Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore specifications
│
├── src/                        # Core algorithmic modules
│   ├── __init__.py
│   ├── segmentation/           # AI feature & boundary segmentation
│   │   ├── __init__.py
│   │   └── inference.py
│   ├── geometry/               # Boundary extraction & geometric processing
│   │   ├── __init__.py
│   │   └── boundary.py
│   ├── vectorization/          # Polygon generation & regularization
│   │   ├── __init__.py
│   │   └── polygons.py
│   ├── change_detection/       # Historical comparison & encroachment flagging
│   │   ├── __init__.py
│   │   └── compare.py
│   └── utils/                  # Geospatial helpers, formatters, GeoJSON export
│       ├── __init__.py
│       └── helpers.py
│
├── data/
│   ├── input/                  # Raw input drone/aerial images
│   ├── cadastral/              # Historical cadastral survey GeoJSON/records
│   └── demo/                   # Demo benchmark sample datasets
│
├── models/                     # Model weights & configurations
├── outputs/                    # Exported GeoJSON, maps, and reports
└── tests/                      # Automated smoke & unit tests
    ├── __init__.py
    └── test_smoke.py
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/vipinsgit-hub/sih-project.git
cd sih-project
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Smoke Tests
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
- [ ] **Milestone 2**: Image upload & preprocessing pipeline.
- [ ] **Milestone 3**: AI-based feature segmentation & boundary extraction.
- [ ] **Milestone 4**: Parcel vectorization, polygon regularization & topology validation.
- [ ] **Milestone 5**: Interactive GIS visualization with Folium.
- [ ] **Milestone 6**: Historical cadastral comparison & encroachment detection.
- [ ] **Milestone 7**: Surveyor verification interface & GeoJSON export.
- [ ] **Milestone 8**: Final testing & demo dataset integration.

---

## 📄 License & Disclaimer

This prototype is built strictly as a technical proof-of-concept for the Smart India Hackathon. It is designed to serve as an **AI-assisted tool for certified surveyors** rather than independently generating legally binding cadastral land titles.
