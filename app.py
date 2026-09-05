"""
SIH Cadastral AI Prototype
Milestone 7: Final Hackathon Dashboard & One-Click Demo Mode
"""
import io
import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st
from PIL import Image

from src import __version__
from src.utils.image_processing import (
    load_image,
    get_image_metadata,
    preprocess_for_model,
    ImageProcessingError,
)
from src.utils.helpers import (
    encroachments_to_geojson_dict,
    format_area_sqm,
)
from src.segmentation.inference import (
    load_segmentation_model,
    segment_image,
    create_segmentation_overlay,
    DEFAULT_MODEL_ID,
)
from src.geometry.boundary import (
    extract_boundaries,
    render_boundary_overlay,
    DEFAULT_PARCEL_FEATURE_CLASSES,
)
from src.vectorization.polygons import (
    generate_candidate_parcels,
    parcels_to_geojson_dict,
    render_parcel_overlay,
)
from src.geospatial.cadastral import (
    load_cadastral_geojson,
    get_cadastral_statistics,
    CadastralDataError,
)
from src.geospatial.comparison import (
    compare_cadastral_vs_candidate_parcels,
)
from src.change_detection.compare import (
    detect_cadastral_changes,
    calculate_change_metrics,
    classify_discrepancy_and_risk,
)
from src.visualization.map import (
    render_gis_comparison_map,
    render_parcel_detail_comparison,
)

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="CADASTRAL AI — Decision Support Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Professional Enterprise Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 0.1rem;
    }
    .tagline {
        font-size: 0.92rem;
        color: #64748B;
        font-style: italic;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.5rem;
    }
    .kpi-title {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .kpi-value {
        font-size: 1.45rem;
        color: #0F172A;
        font-weight: 800;
        margin-top: 0.1rem;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94A3B8;
    }
    .disclaimer-banner {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1.2rem;
        font-size: 0.86rem;
        color: #92400E;
        line-height: 1.4;
    }
    .legend-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .badge-cadastral {
        background-color: #DBEAFE;
        color: #1E40AF;
        border: 1px solid #93C5FD;
    }
    .badge-candidate {
        background-color: #CFFAFE;
        color: #0E7490;
        border: 1px solid #67E8F9;
    }
    .badge-encroachment {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached AI Model Resource
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model(model_id: str = DEFAULT_MODEL_ID):
    """Load and cache the semantic segmentation model bundle."""
    return load_segmentation_model(model_id)

# ---------------------------------------------------------
# Sidebar Configuration & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/map-marker.png", width=52)
    st.title("Cadastral AI")
    st.caption(f"Decision Support System v{__version__}")
    st.markdown("---")

    st.markdown("### ⚙️ Spatial Analysis Rules")
    match_threshold = st.slider(
        "Match Overlap Threshold (IoU %)",
        min_value=70,
        max_value=95,
        value=85,
        step=5,
        help="Minimum spatial overlap to classify parcel as MATCH (Risk: LOW)"
    ) / 100.0

    minor_threshold = st.slider(
        "Minor Mismatch Threshold (IoU %)",
        min_value=40,
        max_value=75,
        value=60,
        step=5,
        help="Spatial overlap for MINOR BOUNDARY MISMATCH (Risk: MEDIUM)"
    ) / 100.0

    encroach_threshold = st.slider(
        "Encroachment Protrusion Threshold (%)",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="Proportion of candidate polygon extending outside cadastral deed line to trigger potential encroachment alert (Risk: HIGH)"
    ) / 100.0

    st.markdown("---")
    st.markdown("### 🗺️ Map Layer Toggles")
    show_cad_layer = st.checkbox("Layer 1: Cadastral Reference (Blue)", value=True)
    show_cand_layer = st.checkbox("Layer 2: AI Candidate Parcels (Cyan)", value=True)
    show_discrepancy_layer = st.checkbox("Layer 3: Potential Encroachment (Red)", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ Dataset Reference")
    st.caption("Benchmark Data: Synthetic Cadastral Survey Records (P-001 to P-005)")
    st.caption("Coordinate System: Normalized Local Coordinate Space")
    st.caption("Smart India Hackathon Prototype")

# ---------------------------------------------------------
# Application Header & Disclaimer
# ---------------------------------------------------------
st.markdown('<div class="main-header">CADASTRAL AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Assisted Land Boundary Monitoring & Change Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">"From aerial imagery to surveyor-ready spatial intelligence"</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    <strong>⚖️ Decision-Support Notice:</strong>
    This prototype identifies potential spatial discrepancies between historical cadastral records and current AI-extracted physical boundaries.
    Flagged anomalies represent <strong>Potential Encroachments & Discrepancies</strong> designed to guide on-site surveyor inspection and do <strong>not</strong> constitute legally confirmed title infringements.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Input Section & One-Click Execution Controls
# ---------------------------------------------------------
input_card = st.container()
with input_card:
    col_upload, col_demo_btn, col_exec_btn = st.columns([2, 1, 1.2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Aerial/Drone Survey Photo",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="High-resolution survey imagery"
        )
    with col_demo_btn:
        st.write("")
        st.write("")
        use_demo = st.button("📁 Load Demo Aerial Survey", use_container_width=True)
    with col_exec_btn:
        st.write("")
        st.write("")
        run_full_pipeline = st.button("⚡ RUN COMPLETE AI ANALYSIS", type="primary", use_container_width=True)

# Resolve Image Source
image_bytes = None
filename = None
file_size = None
demo_image_path = os.path.join("data", "demo", "sample_drone_aerial.png")
demo_cadastral_path = os.path.join("data", "cadastral", "demo_cadastral.geojson")

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    file_size = uploaded_file.size
elif (use_demo or "loaded_image" in st.session_state) and os.path.exists(demo_image_path):
    with open(demo_image_path, "rb") as f:
        image_bytes = f.read()
    filename = "sample_drone_aerial.png"
    file_size = len(image_bytes)

# Initialize Session State Stores
if "surveyor_statuses" not in st.session_state:
    st.session_state["surveyor_statuses"] = {}
if "surveyor_notes" not in st.session_state:
    st.session_state["surveyor_notes"] = {}
if "selected_inspect_id" not in st.session_state:
    st.session_state["selected_inspect_id"] = "P-003"

# ---------------------------------------------------------
# Pipeline Execution Engine
# ---------------------------------------------------------
if image_bytes is not None:
    try:
        loaded_image = load_image(image_bytes)
        metadata = get_image_metadata(loaded_image, filename=filename, file_size_bytes=file_size)
        st.session_state["loaded_image"] = loaded_image
        st.session_state["image_metadata"] = metadata

        # Execute Full Pipeline if triggered or cached
        if run_full_pipeline or "pipeline_complete" not in st.session_state:
            with st.status("⚡ Executing End-to-End Cadastral AI Analysis...", expanded=True) as status_box:
                # Step 1: Preprocessing
                st.write("1️⃣ Ingesting & normalizing survey imagery...")
                model_bundle = get_cached_model(DEFAULT_MODEL_ID)
                
                # Step 2: Semantic Segmentation
                st.write("2️⃣ Performing semantic scene parsing with SegFormer Transformer...")
                seg_result = segment_image(loaded_image, model_bundle)
                st.session_state["seg_result"] = seg_result
                
                # Step 3: Boundary & Regularization
                st.write("3️⃣ Extracting morphological contours & regularizing candidate parcel polygons...")
                boundary_data = extract_boundaries(
                    seg_result["mask"],
                    target_class_ids=DEFAULT_PARCEL_FEATURE_CLASSES,
                    min_area=200.0,
                    apply_morphology=True
                )
                parcel_data = generate_candidate_parcels(
                    boundary_data["contours"],
                    tolerance=2.0,
                    min_area=200.0,
                    id_prefix="C-"
                )
                st.session_state["boundary_data"] = boundary_data
                st.session_state["parcel_data"] = parcel_data
                
                # Step 4: Cadastral Reference Ingestion
                st.write("4️⃣ Ingesting cadastral reference survey records...")
                cad_parcels = load_cadastral_geojson(demo_cadastral_path)
                st.session_state["cad_parcels"] = cad_parcels
                
                # Step 5 & 6: Spatial Topology & Change Detection
                st.write("5️⃣ Executing spatial topology comparison, area deltas & potential encroachment detection...")
                change_data = detect_cadastral_changes(
                    cad_parcels,
                    parcel_data["parcels"],
                    match_iou_threshold=match_threshold,
                    minor_iou_threshold=minor_threshold,
                    encroachment_threshold=encroach_threshold,
                )
                st.session_state["change_data"] = change_data
                
                # Step 7: Finalizing
                st.write("6️⃣ Prioritizing surveyor verification queue & risk classification...")
                st.session_state["pipeline_complete"] = True
                status_box.update(label="✔ Cadastral AI Analysis Complete", state="complete", expanded=False)

        # Retrieve pipeline data from session state
        change_data = st.session_state["change_data"]
        cad_parcels = st.session_state["cad_parcels"]
        parcel_data = st.session_state["parcel_data"]
        seg_result = st.session_state["seg_result"]
        boundary_data = st.session_state["boundary_data"]
        records = change_data["change_records"]
        queue = change_data["surveyor_queue"]

        # Calculate pending review count
        pending_count = sum(
            1 for r in records
            if st.session_state["surveyor_statuses"].get(r["parcel_id"], r["verification_status"]) == "PENDING SURVEYOR REVIEW"
        )

        # ---------------------------------------------------------
        # Executive KPI Cards
        # ---------------------------------------------------------
        st.markdown("---")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Cadastral Parcels</div>
                <div class="kpi-value">{change_data['total_cadastral']}</div>
                <div class="kpi-sub">Reference Plots</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">AI Candidates</div>
                <div class="kpi-value">{change_data['total_candidates']}</div>
                <div class="kpi-sub">Detected Structures</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Matched Plots</div>
                <div class="kpi-value" style="color:#166534;">{change_data['matches_count']}</div>
                <div class="kpi-sub">IoU ≥ {int(match_threshold*100)}%</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Discrepancies</div>
                <div class="kpi-value" style="color:#854D0E;">{change_data['minor_discrepancies_count'] + change_data['significant_discrepancies_count']}</div>
                <div class="kpi-sub">Boundary Variances</div>
            </div>""", unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">🚨 Encroachments</div>
                <div class="kpi-value" style="color:#B91C1C;">{change_data['potential_encroachments_count']}</div>
                <div class="kpi-sub">Protrusion Alerts</div>
            </div>""", unsafe_allow_html=True)
        with k6:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Surveyor Queue</div>
                <div class="kpi-value" style="color:#4338CA;">{pending_count}</div>
                <div class="kpi-sub">Pending Review</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ---------------------------------------------------------
        # Main Dashboard Tabs
        # ---------------------------------------------------------
        tab_gis, tab_inspect, tab_diag, tab_export, tab_about = st.tabs([
            "🗺️ Executive GIS & Priority Findings",
            "🔍 Surveyor Inspection & Sign-Off",
            "⚙️ AI Pipeline Diagnostics",
            "📥 Reports & Export",
            "ℹ️ Methodology & Architecture"
        ])

        # ---------------------------------------------------------
        # TAB 1: Executive GIS & Priority Findings
        # ---------------------------------------------------------
        with tab_gis:
            st.markdown("### 🗺️ Multi-Layer Cadastral GIS Map")

            # Legend Badges
            st.markdown("""
            <div style="margin-bottom: 0.8rem;">
                <span class="legend-badge badge-cadastral">■ Blue: Cadastral Reference Boundary</span>
                <span class="legend-badge badge-candidate">■ Cyan: AI Candidate Parcel Boundary</span>
                <span class="legend-badge badge-encroachment">■ Red: Potential Encroachment / Extension Protrusion</span>
            </div>
            """, unsafe_allow_html=True)

            # Map Data Formatting
            comp_map_data = []
            for r in records:
                comp_map_data.append({
                    "cadastral_id": r["parcel_id"],
                    "candidate_id": r["candidate_id"],
                    "encroachment_geometry": r["potential_extension_geom"] if r["potential_encroachment"] else None,
                })

            col_map_view, col_map_ctrl = st.columns([3.5, 1])

            with col_map_ctrl:
                st.markdown("#### 🎯 Plot Highlighter")
                cad_id_options = ["All Parcels"] + [r["parcel_id"] for r in records]
                selected_plot = st.selectbox("Highlight Specific Parcel:", options=cad_id_options, index=0)
                focused_id = None if selected_plot == "All Parcels" else selected_plot
                
                st.markdown("---")
                st.markdown("#### 📊 Risk Distribution")
                st.write(f"🔴 **High Risk:** {change_data['high_risk_count']} plots")
                st.write(f"🟡 **Medium Risk:** {change_data['medium_risk_count']} plots")
                st.write(f"🟢 **Low Risk:** {change_data['low_risk_count']} plots")

            with col_map_view:
                gis_map_img = render_gis_comparison_map(
                    loaded_image,
                    cadastral_parcels=cad_parcels,
                    candidate_parcels=parcel_data["parcels"],
                    comparison_results=comp_map_data,
                    show_cadastral=show_cad_layer,
                    show_candidates=show_cand_layer,
                    show_discrepancies=show_discrepancy_layer,
                    selected_cadastral_id=focused_id
                )
                st.image(
                    gis_map_img,
                    caption="Multi-Layer GIS Map: High-resolution aerial survey overlaid with reference deed vectors, AI physical footprints, and flagged lateral boundary extensions.",
                    use_container_width=True
                )

            st.markdown("---")

            # Priority Findings Table
            st.markdown("### 🚨 PRIORITY FINDINGS (Ranked by Risk Severity)")
            
            table_rows = []
            for r in queue:
                pid = r["parcel_id"]
                curr_status = st.session_state["surveyor_statuses"].get(pid, r["verification_status"])
                risk = r["risk_level"]
                risk_badge = "🔴 HIGH" if risk == "HIGH" else ("🟡 MEDIUM" if risk == "MEDIUM" else "🟢 LOW")
                
                status_label = r["discrepancy_type"]
                status_badge = f"🚨 {status_label}" if r["potential_encroachment"] else (f"✔ {status_label}" if "MATCH" in status_label and "MISMATCH" not in status_label else f"⚠️ {status_label}")

                table_rows.append({
                    "Parcel ID": pid,
                    "AI Candidate": r["candidate_id"],
                    "Status": status_badge,
                    "Risk": risk_badge,
                    "Spatial Overlap (IoU)": f"{r['overlap_percentage']:.1f}%",
                    "Cadastral Area": f"{r['cadastral_area_px']:,.0f} px²",
                    "AI Area": f"{r['candidate_area_px']:,.0f} px²" if r['candidate_area_px'] > 0 else "0",
                    "Potential Extension": f"{r['potential_extension_area_px']:,.0f} px²",
                    "Surveyor Status": curr_status,
                })

            df_findings = pd.DataFrame(table_rows)
            st.dataframe(df_findings, use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # TAB 2: Surveyor Inspection & Sign-Off
        # ---------------------------------------------------------
        with tab_inspect:
            st.markdown("### 🔍 SURVEYOR INSPECTION STATION")
            st.markdown("Deep-dive inspection workstation with high-resolution zoomed overlays and human-in-the-loop verification sign-off.")

            col_sel_p, col_empty = st.columns([2, 2])
            with col_sel_p:
                inspect_pid = st.selectbox(
                    "Select Parcel to Inspect:",
                    options=[r["parcel_id"] for r in queue],
                    index=0,
                    key="inspect_selector"
                )

            target_record = next((r for r in records if r["parcel_id"] == inspect_pid), None)
            if target_record:
                curr_surveyor_status = st.session_state["surveyor_statuses"].get(inspect_pid, target_record["verification_status"])

                col_detail_img, col_detail_meta = st.columns([1.2, 1])

                with col_detail_img:
                    parcel_detail_img = render_parcel_detail_comparison(
                        loaded_image,
                        cadastral_geom=target_record["cadastral_geometry"],
                        candidate_geom=target_record["candidate_geometry"],
                        extension_geom=target_record["potential_extension_geom"],
                        missing_geom=target_record["missing_cadastral_geom"],
                        crop_to_parcel=True,
                        padding=50
                    )
                    st.image(
                        parcel_detail_img,
                        caption=f"Zoom Detail: Parcel {inspect_pid} | Blue = Legal Deed | Cyan = Physical Structure | Red = Potential Protrusion | Orange = Missing Coverage",
                        use_container_width=True
                    )

                with col_detail_meta:
                    st.markdown(f"#### Plot `{inspect_pid}` Analysis Profile")
                    
                    m1, m2 = st.columns(2)
                    with m1:
                        st.write(f"**Matched Candidate:** `{target_record['candidate_id']}`")
                        st.write(f"**Spatial Overlap (IoU):** `{target_record['overlap_percentage']:.1f}%`")
                        st.write(f"**Cadastral Area:** `{target_record['cadastral_area_px']:,.1f} px²`")
                        st.write(f"**Current AI Area:** `{target_record['candidate_area_px']:,.1f} px²`")
                    with m2:
                        st.write(f"**Area Difference:** `{target_record['area_difference_px']:+,.1f} px² ({target_record['area_difference_pct']:.1f}%)`")
                        st.write(f"**Potential Extension Area:** `{target_record['potential_extension_area_px']:,.1f} px² ({target_record['extension_ratio']*100:.1f}%)`")
                        st.write(f"**Prototype Risk:** `{target_record['risk_level']}`")
                        st.write(f"**Discrepancy Category:** `{target_record['discrepancy_type']}`")

                    st.markdown("---")
                    st.markdown(f"**Current Status:** `{curr_surveyor_status}`")

                    st.markdown("##### ✍️ Surveyor Verification Sign-Off")
                    btn_c1, btn_c2, btn_c3 = st.columns(3)
                    with btn_c1:
                        if st.button("🚨 Flag for On-Site Survey", key=f"btn_flag_{inspect_pid}", use_container_width=True):
                            st.session_state["surveyor_statuses"][inspect_pid] = "FLAGGED FOR ON-SITE SURVEY"
                            st.rerun()
                    with btn_c2:
                        if st.button("✓ Mark as Reviewed", key=f"btn_rev_{inspect_pid}", use_container_width=True):
                            st.session_state["surveyor_statuses"][inspect_pid] = "REVIEWED & VERIFIED"
                            st.rerun()
                    with btn_c3:
                        if st.button("🔄 Reset", key=f"btn_rst_{inspect_pid}", use_container_width=True):
                            st.session_state["surveyor_statuses"][inspect_pid] = "PENDING SURVEYOR REVIEW"
                            st.rerun()

                    # Surveyor Notes
                    current_notes = st.session_state["surveyor_notes"].get(inspect_pid, "")
                    surveyor_note = st.text_area(
                        "Surveyor Field Notes / On-Site Instructions:",
                        value=current_notes,
                        key=f"notes_box_{inspect_pid}",
                        height=90
                    )
                    if surveyor_note != current_notes:
                        st.session_state["surveyor_notes"][inspect_pid] = surveyor_note

                    st.caption("📌 Human-in-the-Loop Audit Trail: Status updates and field notes persist across tabs in active session state.")

        # ---------------------------------------------------------
        # TAB 3: AI Pipeline Diagnostics
        # ---------------------------------------------------------
        with tab_diag:
            st.markdown("### ⚙️ AI PIPELINE DIAGNOSTICS & INTERMEDIATE ARTIFACTS")
            st.markdown("Technical inspection view demonstrating each stage of the computer vision and geometry pipeline.")

            diag_t1, diag_t2, diag_t3 = st.tabs([
                "1️⃣ Semantic Feature Map",
                "2️⃣ Contours & Morphology",
                "3️⃣ Regularized Vectors"
            ])

            with diag_t1:
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    sem_overlay = create_segmentation_overlay(loaded_image, seg_result["mask"], alpha=0.45)
                    st.image(sem_overlay, caption="SegFormer-B0 Semantic Feature Overlay", use_container_width=True)
                with col_d2:
                    st.image(seg_result["colored_mask"], caption="150-Class Semantic Mask (ADE20K)", use_container_width=True)
                    st.write(f"**Dominant Class:** `{seg_result['dominant_class']}` ({seg_result['dominant_class_ratio']*100:.1f}%)")

            with diag_t2:
                col_d3, col_d4 = st.columns(2)
                with col_d3:
                    st.image(boundary_data["binary_mask"], caption="Filtered Target Structures (Building, Roof, Wall, Fence)", use_container_width=True)
                with col_d4:
                    bound_overlay = render_boundary_overlay(loaded_image, boundary_data["contours"])
                    st.image(bound_overlay, caption=f"Extracted External Contours ({len(boundary_data['contours'])} detected, {boundary_data['rejected_count']} noise filtered)", use_container_width=True)

            with diag_t3:
                col_d5, col_d6 = st.columns(2)
                with col_d5:
                    poly_overlay = render_parcel_overlay(loaded_image, parcel_data["parcels"])
                    st.image(poly_overlay, caption=f"Douglas-Peucker Regularized Candidate Polygons ({len(parcel_data['parcels'])} valid)", use_container_width=True)
                with col_d6:
                    poly_metrics = []
                    for p in parcel_data["parcels"]:
                        poly_metrics.append({
                            "Candidate ID": p["parcel_id"],
                            "Area (px²)": f"{p['pixel_area']:,.1f}",
                            "Perimeter (px)": f"{p['pixel_perimeter']:,.1f}",
                            "Vertices": p["vertex_count"],
                            "Solidity": p["solidity"],
                            "Status": p["status"],
                        })
                    st.dataframe(pd.DataFrame(poly_metrics), use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # TAB 4: Reports & Export
        # ---------------------------------------------------------
        with tab_export:
            st.markdown("### 📥 REPORTS & SURVEYOR AUDIT EXPORT")
            st.markdown("Generate standard OGC GeoJSON vector files and CSV audit logs for downstream municipal GIS and field surveyor teams.")

            # 1. Candidate Parcels GeoJSON
            candidate_geojson_dict = parcels_to_geojson_dict(parcel_data["parcels"], image_metadata=metadata)
            candidate_geojson_str = json.dumps(candidate_geojson_dict, indent=2)

            # 2. Potential Encroachment GeoJSON
            encroachment_geojson_dict = encroachments_to_geojson_dict(records, metadata=metadata)
            encroachment_geojson_str = json.dumps(encroachment_geojson_dict, indent=2)

            # 3. CSV Audit Report
            audit_rows = []
            for r in queue:
                pid = r["parcel_id"]
                curr_status = st.session_state["surveyor_statuses"].get(pid, r["verification_status"])
                notes = st.session_state["surveyor_notes"].get(pid, "None")
                audit_rows.append({
                    "Cadastral_ID": pid,
                    "Candidate_ID": r["candidate_id"],
                    "IoU_Overlap_Pct": r["overlap_percentage"],
                    "Cadastral_Area_px": r["cadastral_area_px"],
                    "Candidate_Area_px": r["candidate_area_px"],
                    "Area_Difference_px": r["area_difference_px"],
                    "Area_Difference_Pct": r["area_difference_pct"],
                    "Potential_Extension_Area_px": r["potential_extension_area_px"],
                    "Discrepancy_Type": r["discrepancy_type"],
                    "Prototype_Risk_Level": r["risk_level"],
                    "Surveyor_Verification_Status": curr_status,
                    "Surveyor_Notes": notes,
                })
            df_audit = pd.DataFrame(audit_rows)
            csv_str = df_audit.to_csv(index=False)

            exp_c1, exp_c2, exp_c3 = st.columns(3)
            with exp_c1:
                st.download_button(
                    label="📄 Download Candidate Parcels (GeoJSON)",
                    data=candidate_geojson_str,
                    file_name="candidate_parcels.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )
                st.caption(f"Contains {len(parcel_data['parcels'])} AI candidate parcel polygons.")

            with exp_c2:
                st.download_button(
                    label="🚨 Download Encroachments (GeoJSON)",
                    data=encroachment_geojson_str,
                    file_name="potential_encroachments.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )
                st.caption(f"Contains {len(encroachment_geojson_dict['features'])} flagged lateral protrusion polygons.")

            with exp_c3:
                st.download_button(
                    label="📊 Download Surveyor Audit Report (CSV)",
                    data=csv_str,
                    file_name="cadastral_surveyor_audit_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption("Complete table of risk tiers, IoU overlap, and surveyor field notes.")

            st.markdown("---")
            st.markdown("#### 📋 Live Surveyor Audit Log Preview")
            st.dataframe(df_audit, use_container_width=True, hide_index=True)

        # ---------------------------------------------------------
        # TAB 5: Methodology & Architecture
        # ---------------------------------------------------------
        with tab_about:
            st.markdown("### ℹ️ METHODOLOGY & MATHEMATICAL FRAMEWORK")
            st.markdown("""
            #### 1. End-to-End Processing Chain
            ```
            Aerial Survey Image
                    ↓
            Standardized RGB Normalization & Tensor Preprocessing
                    ↓
            AI Semantic Scene Segmentation (SegFormer Transformer ADE20K)
                    ↓
            Structural Feature Filtering (Buildings, Roofs, Walls, Fences)
                    ↓
            Morphological Denoising & Contour Extraction (OpenCV)
                    ↓
            Douglas-Peucker Regularization & Shapely Topology Validation
                    ↓
            Cadastral Reference Ingestion (GeoJSON)
                    ↓
            Spatial Topology Intersection & IoU Geometric Alignment
                    ↓
            Change Detection & Potential Encroachment Isolation (Candidate ∖ Cadastral)
                    ↓
            Risk Classification (LOW / MEDIUM / HIGH)
                    ↓
            Human-in-the-Loop Surveyor Inspection & Sign-Off Station
            ```

            #### 2. Spatial Overlap Metric (IoU)
            $$\\text{Spatial Overlap Score (IoU)} = \\frac{\\text{Area}(\\text{Cadastral Reference} \\cap \\text{AI Candidate})}{\\text{Area}(\\text{Cadastral Reference} \\cup \\text{AI Candidate})}$$

            #### 3. Potential Lateral Encroachment Formulation
            $$\\text{Potential Extension Area} = \\text{Area}(\\text{AI Candidate} \\setminus \\text{Cadastral Reference})$$
            If $\\frac{\\text{Potential Extension Area}}{\\text{Cadastral Area}} \\ge 10\\%$ and $\\text{IoU} > 15\\%$, the parcel is flagged as **POTENTIAL ENCROACHMENT (HIGH RISK)**.

            #### 4. Limitations & Real-World Georeferencing Roadmap
            - **Current Coordinate Space**: Standardized local pixel coordinate space (`px²`, `px`).
            - **Synthetic Benchmark Dataset**: Evaluated on synthetic reference survey records.
            - **Georeferencing Roadmap**: Designed with modular decoupling (`src/geospatial/`, `src/change_detection/`) ready to ingest georeferenced GeoTIFF imagery (EPSG:4326 / UTM projected coordinates) and official state cadastral Shapefiles in future deployment.
            """)

    except ImageProcessingError as e:
        st.error(f"⚠️ Image Error: {str(e)}")
    except CadastralDataError as e:
        st.error(f"⚠️ Cadastral Data Error: {str(e)}")
    except Exception as e:
        st.error(f"⚠️ Pipeline Error: {str(e)}")
else:
    st.info("👆 Click **'📁 Load Demo Aerial Survey'** or upload an aerial survey image above, then click **'⚡ RUN COMPLETE AI ANALYSIS'** to begin.")
