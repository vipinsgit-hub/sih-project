# SIH Cadastral AI Prototype — Hackathon Demo Checklist

> **Problem Statement**: AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction Using Drone Imagery  
> **Repository**: [https://github.com/vipingsit-hub/sih-project](https://github.com/vipingsit-hub/sih-project)  
> **Prototype Role**: AI-assisted decision-support system for land administration, municipal urban planning, and revenue authority field surveyors.

---

## 1. Pre-Demo Preparation (Offline Safety Check)

Before presenting to the judges:
1. Ensure Python 3.10+ virtual environment is activated:
   ```bash
   .venv\Scripts\activate
   ```
2. Verify all dependencies and cached segmentation model weights are ready:
   ```bash
   python -m unittest discover -s tests
   ```
   *(All 43 unit tests must pass with `OK`.)*
3. Ensure no external internet connection or live API keys are required during the demo — the prototype runs 100% locally and deterministically.

---

## 2. Starting the Application

Launch the Streamlit executive dashboard locally:
```bash
streamlit run app.py
```
- The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 3. Loading the Benchmark Demo Image

1. Click the prominent **`📁 LOAD DEMO AERIAL SURVEY`** button at the top of the dashboard.
2. Observe the benchmark aerial image (512×512 px ortho-mosaic) load instantly.

---

## 4. Running the Complete AI Pipeline

1. Click the primary button: **`⚡ RUN COMPLETE AI & CADASTRAL PIPELINE`**.
2. The entire 5-stage pipeline executes in **< 1.5 seconds**:
   - **Stage 1**: Standardized RGB Normalization & Tensor Preprocessing.
   - **Stage 2**: AI Semantic Feature Segmentation (SegFormer Transformer ADE20K).
   - **Stage 3**: Contour & Boundary Vectorization with Douglas-Peucker Polygon Regularization.
   - **Stage 4**: Cadastral Deed Overlay & Spatial Topology Intersection (IoU, Area Deltas).
   - **Stage 5**: Multi-Criteria Risk Classification & Priority Findings Ranking.
3. Observe the top **Executive Overview KPI Cards** populate:
   - **Total Parcels**: `5` (Cadastral Deed Records)
   - **AI Candidates**: `4` (Detected Footprints)
   - **Verified Matches**: `1` (`P-001`, Low Risk)
   - **High-Risk Findings**: `3` (`P-002`, `P-003`, `P-004`)
   - **Surveyor Review**: `5` (Pending Field Sign-Off)

---

## 5. Recommended Parcel for Demonstration: `P-003`

Highlight parcel **`P-003`** to the judges as the primary showcase of AI change detection:
1. In the **Priority Findings** table, point out `P-003`:
   - **Discrepancy Status**: `🚨 POTENTIAL ENCROACHMENT`
   - **Risk Level**: `🔴 HIGH`
   - **Intersection over Union (IoU)**: `~42.9%`
   - **Potential Extension / Protrusion Area**: `~7,500 px²`
2. In the **Parcel Selector** dropdown, select `P-003` to isolate its bounding geometry.

---

## 6. Demonstrating the Tabbed Visual Analysis

Navigate across the 5 structured tabs:
- **Tab 1: 🗺️ Executive GIS Dashboard**:
  - Point out the 3 GIS layers:
    - **Blue Outline**: Cadastral Reference Deed Boundary
    - **Cyan Outline**: AI Candidate Parcel Footprint
    - **Crimson Protrusion**: Detected Encroachment / Lateral Boundary Extension
- **Tab 2: 🔍 Surveyor Inspection**:
  - Deep-dive zoom detail for selected parcel (`P-003`).
  - Side-by-side comparison metrics (Overlap %, Deed Area, AI Area, Area Difference).
- **Tab 3: 🤖 AI Pipeline**:
  - 3-panel diagnostic pipeline: Semantic Feature Overlay $\rightarrow$ Morphological Contours $\rightarrow$ Douglas-Peucker Regularized Vectors.
- **Tab 4: 📥 Reports & Export**:
  - Candidate Parcel GeoJSON, Encroachment GeoJSON, and CSV Audit Report downloads.
- **Tab 5: ℹ️ Methodology**:
  - Mathematical formulation of IoU, extension geometry, and future GeoTIFF / GSD georeferencing roadmap.

---

## 7. Demonstrating Surveyor Inspection & Sign-Off

1. Switch to **Tab 2: 🔍 Surveyor Inspection**.
2. Explain the governance principle: *"AI flags candidates; certified revenue surveyors retain legal decision authority."*
3. Type field notes into **Surveyor Field Notes / On-Site Instructions**:
   ```
   Flagged for boundary wall inspection
   ```
4. Click **`🚨 Flag for On-Site Survey`** or **`✓ Mark as Reviewed`**.
5. Observe the status update in real time and persist across all dashboard views.

---

## 8. Exporting Audit Reports

1. Switch to **Tab 4: 📥 Reports & Export**.
2. Demonstrate instant export capabilities:
   - **📄 Download Candidate Parcels (GeoJSON)**: For QGIS / ArcGIS ingestion.
   - **🚨 Download Encroachments (GeoJSON)**: Flagged lateral protrusion polygons.
   - **📊 Download Surveyor Audit Report (CSV)**: Complete audit log with surveyor notes.

---

## 9. Resetting the Application

- Click **`🔄 Reset Analysis`** in the left sidebar to clear all session state and return the dashboard to its initial clean state for the next demonstration.
