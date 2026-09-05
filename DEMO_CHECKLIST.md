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

1. On the left sidebar, navigate to **1. Drone Aerial Image Input**.
2. Click the **📁 Load Demo Aerial Survey** button.
3. Observe the benchmark aerial image (512×512 px ortho-mosaic) render instantly with:
   - File details (Dimensions, Channels, Aspect Ratio, Color Stats).
   - RGB Channel Histograms and CLAHE Enhanced Preview.

---

## 4. Running the Complete AI Pipeline

1. In the sidebar, click the prominent **⚡ RUN COMPLETE AI ANALYSIS** button.
2. The entire 5-stage pipeline executes in **< 1.5 seconds**:
   - **Stage 1**: Contrast Limited Adaptive Histogram Equalization (CLAHE).
   - **Stage 2**: AI Semantic Feature Segmentation (DeepLabV3+ / U-Net / Fallback).
   - **Stage 3**: Contour & Boundary Vectorization with Douglas-Peucker Polygon Regularization.
   - **Stage 4**: Cadastral Overlay & Spatial Discrepancy Analysis (IoU, Centroid Shift, Area Deviation).
   - **Stage 5**: Multi-Criteria Risk Classification & Priority Findings Ranking.
3. Observe the top **Executive Summary KPI Cards** populate:
   - **Total Cadastral Parcels Analyzed** (5)
   - **Potential Encroachments Detected** (2 High Risk)
   - **Total Potential Encroachment Area** (~12,000 px²)
   - **High Risk Alerts** (3)

---

## 5. Recommended Parcel for Demonstration: `P-003`

Highlight parcel **`P-003`** to the judges as the primary showcase of AI change detection:
1. In the **Priority Findings** table, point out `P-003`:
   - **Status**: `POTENTIAL ENCROACHMENT`
   - **Risk Level**: `HIGH`
   - **Intersection over Union (IoU)**: `~42.9%`
   - **Potential Encroachment Protrusion Area**: `~7,500 px²`
2. Select `P-003` in the **Select Parcel to Inspect** dropdown.

---

## 6. Demonstrating the Multi-Layer GIS Comparison

Navigate across the interactive tabs:
- **Tab 1: 🗺️ Cadastral GIS Map**:
  - Toggle **Layer 1 (Cadastral Records - Blue Outline)** vs **Layer 2 (AI-Derived Footprints - Green Fill)**.
  - Inspect **Layer 3 (Encroachment Overlay - Bright Red/Magenta Protrusion)** clearly highlighting where physical ground construction extends beyond the official cadastral boundary.
- **Tab 2: 📊 Discrepancy & Encroachment Analysis**:
  - Point out the metrics: Centroid Shift (px), Area Delta %, IoU overlap, and Exact Spatial Coordinates.
- **Tab 3: 🔍 AI Pipeline Diagnostics**:
  - Show the 4-panel intermediate diagnostic view: Preprocessed Image $\rightarrow$ Segmentation Mask $\rightarrow$ Boundary Overlay $\rightarrow$ Regularized Candidate Polygons.

---

## 7. Demonstrating Surveyor Inspection & Sign-Off

1. Switch to **Tab 4: 📝 Surveyor Inspection & Sign-Off**.
2. Explain the governance principle: *"AI flags candidates; certified surveyors retain final authority."*
3. Type field observations into the **Surveyor Field Observations** text area:
   ```
   Confirmed northern structural extension encroaches into public right-of-way. Dispatched field rover team for DGPS ground survey.
   ```
4. Click **🚩 Flag for On-Site Field Survey** or **✅ Mark Survey Reviewed / Verified**.
5. Observe the live audit log update with the timestamp, decision, and surveyor notes.

---

## 8. Exporting Audit Reports

1. Scroll down to the **📥 Export Audit Package & Reports** section.
2. Demonstrate instant export capabilities:
   - **Download GeoJSON**: For direct import into QGIS, ArcGIS, or municipal spatial databases.
   - **Download CSV Audit Report**: For administrative records, legal review, and revenue workflow integration.

---

## 9. Resetting the Application

- Click the **🔄 Reset Analysis** button in the sidebar to clear all session state and return the dashboard to its initial clean state for the next demonstration.
