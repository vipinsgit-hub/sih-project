"""
SIH Cadastral AI Prototype
Milestone 6: Change Detection & Potential Encroachment Analysis
"""
import json
import os
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
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CadastralAI - Change Detection & Encroachment Analysis",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.15rem;
        color: #0F172A;
        font-weight: 700;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: #92400E;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
        border: 1px solid #FCA5A5;
    }
    .risk-medium {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid #FDE047;
    }
    .risk-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid #86EFAC;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model(model_id: str = DEFAULT_MODEL_ID):
    """Load and cache the semantic segmentation model bundle."""
    return load_segmentation_model(model_id)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/map-marker.png", width=56)
    st.title("CadastralAI")
    st.caption(f"Prototype Engine v{__version__}")
    st.markdown("---")

    st.markdown("### 🛠️ Active Milestone")
    st.markdown("**Milestone 6**: Change Detection & Potential Encroachment Analysis")
    
    st.markdown("---")
    st.markdown("### ⚙️ Discrepancy & Risk Rules")
    match_threshold = st.slider(
        "Match IoU Threshold (%)",
        min_value=70,
        max_value=95,
        value=85,
        step=5,
        help="Spatial overlap required to classify as MATCH (Risk: LOW)"
    ) / 100.0

    minor_threshold = st.slider(
        "Minor Mismatch IoU Threshold (%)",
        min_value=40,
        max_value=75,
        value=60,
        step=5,
        help="Spatial overlap for MINOR BOUNDARY MISMATCH (Risk: MEDIUM)"
    ) / 100.0

    encroach_threshold = st.slider(
        "Potential Encroachment Protrusion (%)",
        min_value=5,
        max_value=40,
        value=10,
        step=5,
        help="Proportion of candidate polygon extending outside cadastral boundary to trigger potential encroachment (Risk: HIGH)"
    ) / 100.0

    st.markdown("---")
    st.markdown("### 🗺️ GIS Layer Visibility")
    show_cad_layer = st.checkbox("Layer 1: Cadastral Reference (Blue)", value=True)
    show_cand_layer = st.checkbox("Layer 2: AI Candidate Parcels (Cyan)", value=True)
    show_discrepancy_layer = st.checkbox("Layer 3: Potential Encroachment / Extension (Red)", value=True)

    st.markdown("---")
    st.caption("Smart India Hackathon Prototype | Team CadastralAI")

# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">Cadastral AI: Change Detection & Encroachment Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Assisted Spatial Discrepancy Detection, Geometric Risk Classification & Surveyor Verification Queue</div>', unsafe_allow_html=True)

# Technical & Legal Notice
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Disclaimer & Legal Decision-Support Notice:</strong>
    The system identifies potential spatial discrepancies between existing cadastral reference geometry and AI-derived candidate geometry.
    It <strong>does not establish legal property boundaries or confirm encroachment</strong>. Final determination requires qualified surveyor and/or authorized municipal authority verification.
</div>
""", unsafe_allow_html=True)

# Main Workspace Tabs
tab_pipeline, tab_workflow = st.tabs(["🚀 End-to-End Pipeline & Analysis", "ℹ️ Methodology & Architecture"])

with tab_pipeline:
    # ---------------------------------------------------------
    # Stage 1: Aerial Survey Input
    # ---------------------------------------------------------
    st.subheader("1. Aerial / Drone Survey Input")
    
    col_input, col_sample = st.columns([3, 1])
    with col_input:
        uploaded_file = st.file_uploader(
            "Upload aerial survey imagery (JPG, PNG, TIFF)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="High-resolution survey imagery"
        )
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button("📁 Load Demo Aerial Survey", use_container_width=True)

    # Determine image source
    image_bytes = None
    filename = None
    file_size = None
    demo_image_path = os.path.join("data", "demo", "sample_drone_aerial.png")

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        file_size = uploaded_file.size
    elif use_sample and os.path.exists(demo_image_path):
        with open(demo_image_path, "rb") as f:
            image_bytes = f.read()
        filename = "sample_drone_aerial.png"
        file_size = len(image_bytes)

    if image_bytes is not None:
        try:
            # 1. Safe loading & metadata
            loaded_image = load_image(image_bytes)
            metadata = get_image_metadata(loaded_image, filename=filename, file_size_bytes=file_size)

            col_meta_1, col_meta_2, col_meta_3, col_meta_4 = st.columns(4)
            with col_meta_1:
                st.markdown(f"""<div class="metric-box"><div class="metric-title">Filename</div><div class="metric-value" style="font-size:0.95rem;">{metadata['filename']}</div></div>""", unsafe_allow_html=True)
            with col_meta_2:
                st.markdown(f"""<div class="metric-box"><div class="metric-title">Resolution</div><div class="metric-value">{metadata['resolution']}</div></div>""", unsafe_allow_html=True)
            with col_meta_3:
                st.markdown(f"""<div class="metric-box"><div class="metric-title">Channels</div><div class="metric-value">{metadata['channels']} ({metadata['mode']})</div></div>""", unsafe_allow_html=True)
            with col_meta_4:
                st.markdown(f"""<div class="metric-box"><div class="metric-title">File Size</div><div class="metric-value">{metadata.get('file_size_formatted', 'N/A')}</div></div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ---------------------------------------------------------
            # Stage 2: AI Feature Segmentation
            # ---------------------------------------------------------
            st.subheader("2. AI Semantic Feature Segmentation")
            col_btn, col_info = st.columns([1, 2])
            with col_btn:
                run_ai = st.button("⚡ Run AI Feature Segmentation", type="primary", use_container_width=True)

            if run_ai or "seg_result" in st.session_state:
                if run_ai:
                    with st.spinner("🧠 Computing semantic scene parsing with SegFormer Transformer..."):
                        model_bundle = get_cached_model(DEFAULT_MODEL_ID)
                        seg_result = segment_image(loaded_image, model_bundle)
                        st.session_state["seg_result"] = seg_result
                        st.session_state.pop("parcel_result", None)
                        st.session_state.pop("change_result", None)
                else:
                    seg_result = st.session_state["seg_result"]

                # Quick Diagnostics View
                with st.expander("👁️ View AI Semantic Segmentation Diagnostics", expanded=False):
                    v_tab1, v_tab2 = st.tabs(["Semantic Overlay", "Color Mask"])
                    with v_tab1:
                        dynamic_overlay = create_segmentation_overlay(loaded_image, seg_result["mask"], alpha=0.40)
                        st.image(dynamic_overlay, caption="AI Semantic Feature Map", use_container_width=True)
                    with v_tab2:
                        st.image(seg_result["colored_mask"], caption="Semantic Classes", use_container_width=True)

                st.markdown("---")

                # ---------------------------------------------------------
                # Stage 3: Boundary Extraction & Parcel Vectorization
                # ---------------------------------------------------------
                st.subheader("3. Candidate Parcel Boundary Extraction & Vectorization")
                col_poly_btn, col_poly_hint = st.columns([1, 2])
                with col_poly_btn:
                    extract_btn = st.button("📐 Extract Candidate Parcels", type="primary", use_container_width=True)

                if extract_btn or "parcel_result" in st.session_state:
                    if extract_btn:
                        with st.spinner("🔍 Extracting contours and regularizing polygons..."):
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
                            st.session_state["parcel_result"] = True
                            st.session_state.pop("change_result", None)
                    else:
                        boundary_data = st.session_state["boundary_data"]
                        parcel_data = st.session_state["parcel_data"]

                    parcels = parcel_data["parcels"]
                    st.info(f"✔ Extracted {len(parcels)} candidate parcel polygons ({boundary_data['rejected_count']} noise regions filtered).")

                    st.markdown("---")

                    # ---------------------------------------------------------
                    # Stage 4: Cadastral Ingestion & Change Detection
                    # ---------------------------------------------------------
                    st.subheader("4. Change Detection & Potential Encroachment Analysis")

                    demo_cadastral_path = os.path.join("data", "cadastral", "demo_cadastral.geojson")
                    
                    cad_col1, cad_col2 = st.columns([3, 1])
                    with cad_col1:
                        cadastral_source = st.selectbox(
                            "Cadastral Survey Reference Source",
                            options=[f"Demo Cadastral Survey Records ({demo_cadastral_path})"],
                            index=0
                        )
                    with cad_col2:
                        st.write("")
                        st.write("")
                        run_change_btn = st.button("⚡ Run Change & Encroachment Analysis", type="primary", use_container_width=True)

                    # Initialize surveyor verification statuses in session state if absent
                    if "surveyor_statuses" not in st.session_state:
                        st.session_state["surveyor_statuses"] = {}
                    if "surveyor_notes" not in st.session_state:
                        st.session_state["surveyor_notes"] = {}

                    if run_change_btn or "change_result" in st.session_state:
                        if run_change_btn:
                            with st.spinner("🔍 Analyzing geometric discrepancies, area deltas, and risk tiers..."):
                                cad_parcels = load_cadastral_geojson(demo_cadastral_path)
                                change_data = detect_cadastral_changes(
                                    cad_parcels,
                                    parcels,
                                    match_iou_threshold=match_threshold,
                                    minor_iou_threshold=minor_threshold,
                                    encroachment_threshold=encroach_threshold,
                                )
                                st.session_state["cad_parcels"] = cad_parcels
                                st.session_state["change_data"] = change_data
                                st.session_state["change_result"] = True
                        else:
                            cad_parcels = st.session_state["cad_parcels"]
                            change_data = st.session_state["change_data"]

                        records = change_data["change_records"]
                        queue = change_data["surveyor_queue"]

                        # ---------------------------------------------------------
                        # Summary Cards
                        # ---------------------------------------------------------
                        c_m1, c_m2, c_m3, c_m4, c_m5, c_m6 = st.columns(6)
                        with c_m1:
                            st.metric("Cadastral Parcels", change_data["total_cadastral"])
                        with c_m2:
                            st.metric("AI Candidates", change_data["total_candidates"])
                        with c_m3:
                            st.metric("Matches (IoU ≥ 85%)", change_data["matches_count"])
                        with c_m4:
                            st.metric("Discrepancies", change_data["minor_discrepancies_count"] + change_data["significant_discrepancies_count"])
                        with c_m5:
                            st.metric("🚨 Potential Encroachments", change_data["potential_encroachments_count"])
                        with c_m6:
                            st.metric("⚠️ High Risk Tier", change_data["high_risk_count"])

                        st.markdown("---")

                        # ---------------------------------------------------------
                        # Multi-Layer GIS Map
                        # ---------------------------------------------------------
                        st.markdown("### 🗺️ GIS Layered Discrepancy Map")

                        # Format comparison results for GIS map rendering
                        comp_map_data = []
                        for r in records:
                            comp_map_data.append({
                                "cadastral_id": r["parcel_id"],
                                "candidate_id": r["candidate_id"],
                                "encroachment_geometry": r["potential_extension_geom"] if r["potential_encroachment"] else None,
                            })

                        cad_id_options = ["All Parcels"] + [r["parcel_id"] for r in records]
                        selected_plot = st.selectbox("🎯 Highlight Specific Parcel:", options=cad_id_options, index=0)
                        focused_id = None if selected_plot == "All Parcels" else selected_plot

                        gis_map_img = render_gis_comparison_map(
                            loaded_image,
                            cadastral_parcels=cad_parcels,
                            candidate_parcels=parcels,
                            comparison_results=comp_map_data,
                            show_cadastral=show_cad_layer,
                            show_candidates=show_cand_layer,
                            show_discrepancies=show_discrepancy_layer,
                            selected_cadastral_id=focused_id
                        )

                        st.image(
                            gis_map_img,
                            caption="GIS Layered Map: Blue = Cadastral Reference | Cyan = AI Candidate | Red = Potential Encroachment Protrusion",
                            use_container_width=True
                        )

                        st.markdown("---")

                        # ---------------------------------------------------------
                        # Risk & Discrepancy Table
                        # ---------------------------------------------------------
                        st.markdown("### 📊 Cadastral Risk & Discrepancy Registry")

                        table_rows = []
                        for r in queue:
                            pid = r["parcel_id"]
                            curr_status = st.session_state["surveyor_statuses"].get(pid, r["verification_status"])
                            
                            # Risk badge
                            risk = r["risk_level"]
                            if risk == "HIGH":
                                risk_badge = "🔴 HIGH"
                            elif risk == "MEDIUM":
                                risk_badge = "🟡 MEDIUM"
                            else:
                                risk_badge = "🟢 LOW"

                            # Status badge
                            status_label = r["discrepancy_type"]
                            if r["potential_encroachment"]:
                                status_badge = f"🚨 {status_label}"
                            elif "MATCH" in status_label and "MISMATCH" not in status_label:
                                status_badge = f"✔ {status_label}"
                            else:
                                status_badge = f"⚠️ {status_label}"

                            table_rows.append({
                                "Parcel ID": pid,
                                "AI Candidate": r["candidate_id"],
                                "IoU Overlap": f"{r['overlap_percentage']:.1f}%",
                                "Cadastral Area": f"{r['cadastral_area_px']:,.0f} px²",
                                "AI Area": f"{r['candidate_area_px']:,.0f} px²" if r['candidate_area_px'] > 0 else "0",
                                "Area Delta (%)": f"{r['area_difference_pct']:.1f}%",
                                "Potential Extension": f"{r['potential_extension_area_px']:,.0f} px²",
                                "Discrepancy Status": status_badge,
                                "Prototype Risk": risk_badge,
                                "Surveyor Status": curr_status,
                            })

                        df_risk = pd.DataFrame(table_rows)
                        st.dataframe(df_risk, use_container_width=True, hide_index=True)

                        st.markdown("---")

                        # ---------------------------------------------------------
                        # Selected Parcel Detail View & Surveyor Review Actions
                        # ---------------------------------------------------------
                        st.markdown("### 🔍 Parcel Detail Inspector & Surveyor Action Queue")
                        
                        target_pid = st.selectbox(
                            "Select Parcel for Detail Review & Surveyor Sign-Off:",
                            options=[r["parcel_id"] for r in queue],
                            index=0
                        )

                        target_record = next((r for r in records if r["parcel_id"] == target_pid), None)
                        if target_record:
                            curr_surveyor_status = st.session_state["surveyor_statuses"].get(target_pid, target_record["verification_status"])

                            col_detail_map, col_detail_info = st.columns([1, 1])

                            with col_detail_map:
                                # Render single-parcel focused cropped comparison
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
                                    caption=f"Detail Zoom for {target_pid}: Blue = Legal Boundary | Cyan = Current Structure | Red = Potential Protrusion | Orange = Missing Coverage",
                                    use_container_width=True
                                )

                            with col_detail_info:
                                st.markdown(f"#### Parcel `{target_pid}` vs Candidate `{target_record['candidate_id']}`")
                                
                                # Metrics Grid
                                d1, d2 = st.columns(2)
                                with d1:
                                    st.write(f"**Spatial Overlap (IoU):** {target_record['overlap_percentage']:.1f}%")
                                    st.write(f"**Existing Cadastral Area:** {target_record['cadastral_area_px']:,.1f} px²")
                                    st.write(f"**Current Candidate Area:** {target_record['candidate_area_px']:,.1f} px²")
                                with d2:
                                    st.write(f"**Area Difference:** {target_record['area_difference_px']:+,.1f} px² ({target_record['area_difference_pct']:.1f}%)")
                                    st.write(f"**Potential Extension Area:** {target_record['potential_extension_area_px']:,.1f} px² ({target_record['extension_ratio']*100:.1f}%)")
                                    st.write(f"**Prototype Risk Tier:** `{target_record['risk_level']}`")

                                st.write(f"**Discrepancy Category:** `{target_record['discrepancy_type']}`")
                                st.write(f"**Current Surveyor Status:** `{curr_surveyor_status}`")

                                st.markdown("##### ✍️ Surveyor Verification Action")
                                
                                act_c1, act_c2, act_c3 = st.columns(3)
                                with act_c1:
                                    if st.button("✔ Mark Reviewed", key=f"rev_{target_pid}", use_container_width=True):
                                        st.session_state["surveyor_statuses"][target_pid] = "REVIEWED & VERIFIED"
                                        st.rerun()
                                with act_c2:
                                    if st.button("🚨 Flag for On-Site Survey", key=f"flag_{target_pid}", use_container_width=True):
                                        st.session_state["surveyor_statuses"][target_pid] = "FLAGGED FOR ON-SITE SURVEY"
                                        st.rerun()
                                with act_c3:
                                    if st.button("🔄 Reset Status", key=f"rst_{target_pid}", use_container_width=True):
                                        st.session_state["surveyor_statuses"][target_pid] = "PENDING SURVEYOR REVIEW"
                                        st.rerun()

                                # Optional Surveyor Notes
                                current_notes = st.session_state["surveyor_notes"].get(target_pid, "")
                                notes_input = st.text_input("Surveyor Field Notes:", value=current_notes, key=f"note_in_{target_pid}")
                                if notes_input != current_notes:
                                    st.session_state["surveyor_notes"][target_pid] = notes_input

                                st.caption("📌 Human-in-the-loop: Surveyor review records persist in active session state for quality audit trail.")

                        st.success("✔ Milestone 6 Complete: Change detection, potential encroachment analysis, and surveyor verification queue operational.")

        except ImageProcessingError as e:
            st.error(f"⚠️ Image Error: {str(e)}")
        except CadastralDataError as e:
            st.error(f"⚠️ Cadastral Data Error: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ Pipeline Error: {str(e)}")
    else:
        st.info("👆 Upload an aerial survey image or click 'Load Demo Aerial Survey' above to begin.")

with tab_workflow:
    st.markdown("""
    ### 🔬 Change Detection & Encroachment Methodology
    
    1. **Spatial Geometry Matching**:
       - Maps each existing cadastral parcel to candidate AI polygons via spatial intersection.
    2. **Geometric Change Metrics**:
       - **Intersection over Union (IoU)**: Evaluates overall geometric similarity.
       - **Potential Extension Area**: Computes candidate geometry extending outside reference boundary:
         $$\\text{Extension} = \\text{Candidate} \\setminus \\text{Cadastral}$$
       - **Missing Cadastral Area**: Computes legal parcel area not covered by detected structure.
       - **Area Delta (%)**: Quantifies relative structural footprint variance.
    3. **Discrepancy & Risk Classification**:
       - **MATCH (Low Risk)**: IoU $\\ge$ 85% with minimal difference.
       - **MINOR BOUNDARY MISMATCH (Medium Risk)**: 60% $\\le$ IoU < 85%.
       - **SIGNIFICANT BOUNDARY MISMATCH (High Risk)**: IoU < 60%.
       - **POTENTIAL ENCROACHMENT (High Risk)**: Extension area $\\ge$ 10% of cadastral area.
       - **UNMATCHED (Medium Risk)**: Reference parcel with no candidate structure detected.
    4. **Surveyor Verification Queue (Human-in-the-Loop)**:
       - Flags high-risk plots into an actionable verification queue.
       - Allows certified surveyors to inspect deep-dive zoomed overlays, add notes, and sign off as `REVIEWED` or `FLAGGED FOR ON-SITE SURVEY`.
    """)
