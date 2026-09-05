"""
SIH Cadastral AI Prototype
Milestone 5: GIS Visualization & Spatial Cadastral Comparison
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
from src.visualization.map import (
    render_gis_comparison_map,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CadastralAI - GIS Visualization & Comparison",
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
    .status-match {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .status-minor {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .status-mismatch {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .status-encroach {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 700;
        border: 1px solid #FCA5A5;
        font-size: 0.82rem;
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
    st.markdown("**Milestone 5**: GIS Visualization & Cadastral Comparison")
    
    st.markdown("---")
    st.markdown("### ⚙️ Spatial Comparison Rules")
    match_threshold = st.slider(
        "Match IoU Threshold (%)",
        min_value=70,
        max_value=95,
        value=85,
        step=5,
        help="Spatial overlap required to classify as MATCH"
    ) / 100.0

    minor_threshold = st.slider(
        "Minor Mismatch IoU Threshold (%)",
        min_value=40,
        max_value=75,
        value=60,
        step=5,
        help="Spatial overlap for MINOR MISMATCH category"
    ) / 100.0

    encroach_threshold = st.slider(
        "Potential Encroachment Protrusion (%)",
        min_value=5,
        max_value=40,
        value=15,
        step=5,
        help="Proportion of candidate polygon extending outside cadastral boundary to trigger potential encroachment alert"
    ) / 100.0

    st.markdown("---")
    st.markdown("### 🗺️ GIS Layer Visibility")
    show_cad_layer = st.checkbox("Layer 1: Cadastral Reference (Blue)", value=True)
    show_cand_layer = st.checkbox("Layer 2: AI Candidate Parcels (Cyan)", value=True)
    show_discrepancy_layer = st.checkbox("Layer 3: Potential Encroachment / Discrepancy (Red)", value=True)

    st.markdown("---")
    st.caption("Smart India Hackathon Prototype | Team CadastralAI")

# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">AI Cadastral Mapping & GIS Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Urban Parcel Extraction, Cadastral Alignment & Potential Encroachment Detection</div>', unsafe_allow_html=True)

# Technical & Legal Notice
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Technical Distinction & Legal Notice:</strong>
    This prototype detects <em>potential spatial discrepancies</em> between existing cadastral survey geometry and AI-extracted physical structures.
    Flagged anomalies represent <strong>Potential Encroachments / Boundary Discrepancies</strong> intended to guide on-site surveyor inspection and do <strong>not</strong> constitute legally confirmed title infringements.
</div>
""", unsafe_allow_html=True)

# Main Workspace Tabs
tab_pipeline, tab_workflow = st.tabs(["🚀 Cadastral AI Pipeline", "ℹ️ Methodology & Architecture"])

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
                        st.session_state.pop("comp_result", None)
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
                                id_prefix="P-"
                            )
                            st.session_state["boundary_data"] = boundary_data
                            st.session_state["parcel_data"] = parcel_data
                            st.session_state["parcel_result"] = True
                            st.session_state.pop("comp_result", None)
                    else:
                        boundary_data = st.session_state["boundary_data"]
                        parcel_data = st.session_state["parcel_data"]

                    parcels = parcel_data["parcels"]

                    st.info(f"✔ Extracted {len(parcels)} candidate parcel polygons ({boundary_data['rejected_count']} noise regions filtered).")

                    st.markdown("---")

                    # ---------------------------------------------------------
                    # Stage 4: Cadastral Comparison & GIS Visualization
                    # ---------------------------------------------------------
                    st.subheader("4. Historical Cadastral GIS Comparison & Encroachment Detection")

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
                        run_compare_btn = st.button("⚡ Run Spatial Cadastral Comparison", type="primary", use_container_width=True)

                    if run_compare_btn or "comp_result" in st.session_state:
                        if run_compare_btn:
                            with st.spinner("🗺️ Performing spatial topology comparison (IoU, Area Diff, Encroachment Analysis)..."):
                                cad_parcels = load_cadastral_geojson(demo_cadastral_path)
                                comp_data = compare_cadastral_vs_candidate_parcels(
                                    cad_parcels,
                                    parcels,
                                    match_iou_threshold=match_threshold,
                                    minor_iou_threshold=minor_threshold,
                                    encroachment_threshold=encroach_threshold,
                                )
                                st.session_state["cad_parcels"] = cad_parcels
                                st.session_state["comp_data"] = comp_data
                                st.session_state["comp_result"] = True
                        else:
                            cad_parcels = st.session_state["cad_parcels"]
                            comp_data = st.session_state["comp_data"]

                        comp_results = comp_data["comparison_results"]

                        # Comparison Summary Metrics
                        c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
                        with c_m1:
                            st.metric("Cadastral Records", comp_data["total_cadastral"])
                        with c_m2:
                            st.metric("AI Candidate Parcels", comp_data["total_candidates"])
                        with c_m3:
                            st.metric("Matches (IoU ≥ 85%)", comp_data["matches_count"])
                        with c_m4:
                            st.metric("Boundary Mismatches", comp_data["boundary_mismatches_count"] + comp_data["minor_mismatches_count"])
                        with c_m5:
                            st.metric("🚨 Potential Encroachments", comp_data["potential_encroachments_count"])

                        # Multi-Layer GIS Visualization Map
                        st.markdown("### 🗺️ Multi-Layer Cadastral GIS Map")

                        # Optional Parcel Selector for focused inspection
                        cad_id_options = ["All Parcels"] + [c["cadastral_id"] for c in comp_results]
                        selected_plot = st.selectbox("🎯 Highlight Specific Parcel Plot:", options=cad_id_options, index=0)
                        focused_id = None if selected_plot == "All Parcels" else selected_plot

                        gis_map_img = render_gis_comparison_map(
                            loaded_image,
                            cadastral_parcels=cad_parcels,
                            candidate_parcels=parcels,
                            comparison_results=comp_results,
                            show_cadastral=show_cad_layer,
                            show_candidates=show_cand_layer,
                            show_discrepancies=show_discrepancy_layer,
                            selected_cadastral_id=focused_id
                        )

                        st.image(
                            gis_map_img,
                            caption="GIS Layered Map: Blue = Historical Cadastral Record | Cyan = AI Candidate Geometry | Red = Potential Encroachment Region",
                            use_container_width=True
                        )

                        # Detailed Spatial Comparison Table
                        st.markdown("### 📋 Spatial Comparison & Discrepancy Registry")
                        
                        table_data = []
                        for r in comp_results:
                            status_label = r["discrepancy_type"]
                            if r["potential_encroachment"]:
                                formatted_status = f"🚨 {status_label}"
                            elif status_label == "MATCH":
                                formatted_status = f"✔ {status_label}"
                            elif status_label == "MINOR MISMATCH":
                                formatted_status = f"⚠️ {status_label}"
                            else:
                                formatted_status = f"❌ {status_label}"

                            table_data.append({
                                "Cadastral ID": r["cadastral_id"],
                                "Matched AI ID": r["candidate_id"],
                                "Spatial Overlap (IoU)": f"{r['overlap_percentage']:.1f}%",
                                "Cadastral Area (px²)": f"{r['cadastral_area_px']:,.1f}",
                                "AI Area (px²)": f"{r['candidate_area_px']:,.1f}" if r['candidate_area_px'] > 0 else "0.0",
                                "Excess Area (px²)": f"{r['excess_area_px']:,.1f}",
                                "Discrepancy Status": formatted_status,
                                "Land Use": r["land_use"],
                            })

                        df_comp = pd.DataFrame(table_data)
                        st.dataframe(df_comp, use_container_width=True, hide_index=True)

                        # Individual Plot Deep-Dive Inspector
                        if focused_id is not None:
                            target_record = next((r for r in comp_results if r["cadastral_id"] == focused_id), None)
                            if target_record:
                                st.markdown(f"#### 🔍 Deep Inspection: Plot `{focused_id}`")
                                insp_col1, insp_col2 = st.columns(2)
                                with insp_col1:
                                    st.write(f"**Owner Reference:** {target_record['owner_reference']}")
                                    st.write(f"**Land Use Category:** {target_record['land_use']}")
                                    st.write(f"**Spatial Overlap (IoU):** {target_record['overlap_percentage']:.1f}%")
                                    st.write(f"**Area Difference:** {target_record['area_difference_px']:+,.1f} px²")
                                with insp_col2:
                                    st.write(f"**Discrepancy Status:** `{target_record['discrepancy_type']}`")
                                    st.write(f"**Potential Encroachment Alert:** `{'YES - Boundary Protrusion Detected' if target_record['potential_encroachment'] else 'No'}`")
                                    st.write(f"**Protrusion Outside Legal Boundary:** `{target_record['excess_area_px']:,.1f} px² ({target_record['excess_ratio']*100:.1f}%)`")
                                    st.caption("📌 Action Recommended: Flag for certified surveyor field inspection & deed verification.")

                        st.success("✔ Milestone 5 Complete: Multi-layer GIS visualization, spatial IoU comparison, and potential encroachment alerts operational.")

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
    ### 🔬 Spatial Cadastral Comparison Methodology
    
    1. **Cadastral Reference Parsing**: Historical survey GeoJSON records are ingested and validated for topological consistency.
    2. **Spatial Topology Intersect**: Each cadastral plot is mapped to its best candidate AI structure using spatial intersection.
    3. **Intersection over Union (IoU)**: Evaluates geometric alignment:
       $$\\text{IoU} = \\frac{\\text{Area}(\\text{Cadastral} \\cap \\text{AI Candidate})}{\\text{Area}(\\text{Cadastral} \\cup \\text{AI Candidate})}$$
    4. **Potential Encroachment Computation**: Isolates excess geometry extending beyond the legal property line:
       $$\\text{Excess} = \\text{AI Candidate} \\setminus \\text{Cadastral Reference}$$
       If the protrusion exceeds the configured threshold (default: 15%), the parcel is flagged as **POTENTIAL ENCROACHMENT**.
    5. **Multi-Layer GIS Visualization**: Overlays historical deed boundaries (blue), AI detected boundaries (cyan), and potential encroachment zones (red) directly over the original drone survey photo.
    """)
