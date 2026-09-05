"""
SIH Cadastral AI Prototype
Milestone 4: AI Feature Segmentation, Boundary Extraction & Candidate Parcel Vectorization
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

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CadastralAI - Drone Parcel Mapping & Vectorization",
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
    .badge-primary {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: #92400E;
    }
    .section-banner {
        background: #F8FAFC;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        border-left: 4px solid #3B82F6;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
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
    st.markdown("**Milestone 4**: Boundary Extraction & Parcel Generation")
    
    st.markdown("---")
    st.markdown("### 📐 Vectorization Parameters")
    min_parcel_area = st.slider(
        "Min Area Filter (pixels²)",
        min_value=20,
        max_value=2000,
        value=200,
        step=20,
        help="Discard noisy segments smaller than this pixel area"
    )
    simplification_tol = st.slider(
        "Polygon Simplification (Douglas-Peucker)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Tolerance distance for edge regularization"
    )
    overlay_alpha = st.slider(
        "Visual Alpha Opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.40,
        step=0.05
    )

    st.markdown("---")
    st.caption("Smart India Hackathon Prototype | Team CadastralAI")

# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">AI Cadastral Mapping</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Urban Parcel Feature Extraction, Boundary Regularization & Vectorization</div>', unsafe_allow_html=True)

# Disclaimer Box
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Technical Distinction & Prototype Notice:</strong>
    Candidate parcel polygons generated here are <strong>AI-assisted geometric approximations</strong> derived from physical visual features (structures, boundary walls, fences).
    They do <strong>not</strong> represent legally official land titles. Area and perimeter values are measured strictly in <strong>pixel units</strong> (px², px) unless real-world georeferencing/GSD calibration is applied.
</div>
""", unsafe_allow_html=True)

# Main Workspace Tabs
tab_analysis, tab_workflow = st.tabs(["🚀 Cadastral AI Pipeline", "ℹ️ Methodology & Architecture"])

with tab_analysis:
    st.subheader("1. Aerial / Drone Survey Input")
    
    col_input, col_sample = st.columns([3, 1])
    with col_input:
        uploaded_file = st.file_uploader(
            "Upload aerial drone imagery (JPG, PNG, TIFF)",
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
                        st.session_state.pop("parcel_result", None)  # Reset downstream parcel result on re-run
                else:
                    seg_result = st.session_state["seg_result"]

                # Quick Diagnostics View
                with st.expander("👁️ View AI Semantic Segmentation Diagnostics", expanded=False):
                    v_tab1, v_tab2 = st.tabs(["Semantic Overlay", "Color Mask"])
                    with v_tab1:
                        dynamic_overlay = create_segmentation_overlay(
                            loaded_image,
                            seg_result["mask"],
                            alpha=overlay_alpha
                        )
                        st.image(dynamic_overlay, caption="AI Semantic Feature Map", use_container_width=True)
                    with v_tab2:
                        st.image(seg_result["colored_mask"], caption="Semantic Classes", use_container_width=True)

                st.markdown("---")

                # ---------------------------------------------------------
                # Stage 3 & 4: Boundary Extraction & Parcel Vectorization
                # ---------------------------------------------------------
                st.subheader("3. Candidate Parcel Boundary Extraction & Vectorization")
                
                col_poly_btn, col_poly_hint = st.columns([1, 2])
                with col_poly_btn:
                    extract_btn = st.button("📐 Extract Candidate Parcels", type="primary", use_container_width=True)

                if extract_btn or "parcel_result" in st.session_state:
                    if extract_btn:
                        with st.spinner("🔍 Extracting contours, regularizing polygons, and assigning parcel IDs..."):
                            # 1. Boundary extraction
                            boundary_data = extract_boundaries(
                                seg_result["mask"],
                                target_class_ids=DEFAULT_PARCEL_FEATURE_CLASSES,
                                min_area=min_parcel_area,
                                apply_morphology=True
                            )
                            # 2. Polygon regularization & parcel feature generation
                            parcel_data = generate_candidate_parcels(
                                boundary_data["contours"],
                                tolerance=simplification_tol,
                                min_area=min_parcel_area,
                                id_prefix="P-"
                            )
                            # Combine results
                            st.session_state["boundary_data"] = boundary_data
                            st.session_state["parcel_data"] = parcel_data
                            st.session_state["parcel_result"] = True
                    else:
                        boundary_data = st.session_state["boundary_data"]
                        parcel_data = st.session_state["parcel_data"]

                    parcels = parcel_data["parcels"]

                    # Vector Diagnostics Metrics
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    with m_col1:
                        st.metric("Detected Boundary Regions", boundary_data["total_regions_found"])
                    with m_col2:
                        st.metric("Noise Filtered Regions", boundary_data["rejected_count"])
                    with m_col3:
                        st.metric("Valid Candidate Parcels", parcel_data["valid_parcels_count"])
                    with m_col4:
                        st.metric("Geometry Regularization", f"Tol: {simplification_tol} px")

                    # Visual Map Tabs
                    st.markdown("### 🗺️ Parcel Vector Diagnostics")
                    p_tab1, p_tab2, p_tab3 = st.tabs([
                        "Candidate Parcel Polygons",
                        "Extracted Boundary Edges",
                        "Original Aerial Photo"
                    ])

                    with p_tab1:
                        parcel_overlay = render_parcel_overlay(
                            loaded_image,
                            parcels,
                            boundary_color=(0, 220, 255),
                            fill_color=(0, 180, 255),
                            fill_alpha=overlay_alpha,
                            show_labels=True
                        )
                        st.image(parcel_overlay, caption="AI-Assisted Candidate Parcel Boundaries with IDs", use_container_width=True)

                    with p_tab2:
                        b_overlay = render_boundary_overlay(
                            loaded_image,
                            boundary_data["contours"],
                            line_color=(255, 220, 0),
                            thickness=2
                        )
                        st.image(b_overlay, caption="Raw Detected Boundary Contours", use_container_width=True)

                    with p_tab3:
                        st.image(loaded_image, caption=f"Original Survey: {metadata['filename']}", use_container_width=True)

                    # Interactive Candidate Parcel Attributes Table
                    st.markdown("### 📋 Candidate Parcel Registry")
                    
                    if parcels:
                        table_rows = []
                        for p in parcels:
                            table_rows.append({
                                "Parcel ID": p["parcel_id"],
                                "Pixel Area (px²)": f"{p['pixel_area']:,.1f}",
                                "Pixel Perimeter (px)": f"{p['pixel_perimeter']:,.1f}",
                                "Vertices": p["vertex_count"],
                                "Solidity": f"{p['solidity'] * 100:.1f}%",
                                "Centroid (X, Y)": f"({p['centroid_pixel'][0]}, {p['centroid_pixel'][1]})",
                                "Status": p["status"],
                                "Source": p["source"],
                            })
                        df_parcels = pd.DataFrame(table_rows)
                        st.dataframe(df_parcels, use_container_width=True, hide_index=True)

                        # GeoJSON Serialization & Download
                        geojson_data = parcels_to_geojson_dict(parcels, image_metadata=metadata)
                        geojson_str = json.dumps(geojson_data, indent=2)

                        st.download_button(
                            label="📥 Download Candidate Parcels (GeoJSON)",
                            data=geojson_str,
                            file_name=f"candidate_parcels_{metadata['filename'].split('.')[0]}.geojson",
                            mime="application/geo+json",
                            help="Export candidate parcel polygons in standard GeoJSON format"
                        )
                    else:
                        st.warning("No candidate parcels met the minimum area criteria. Try reducing the 'Min Area Filter' in the sidebar.")

                    st.success("✔ Milestone 4 Complete: Boundary extraction, Douglas-Peucker regularization, and GeoJSON polygon generation operational.")

        except ImageProcessingError as e:
            st.error(f"⚠️ Image Error: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ Pipeline Error: {str(e)}")
    else:
        st.info("👆 Upload an aerial survey image or click 'Load Demo Aerial Survey' above to begin.")

with tab_workflow:
    st.markdown("""
    ### 🔄 Geometric Boundary Extraction Pipeline
    
    1. **Feature Mask Filtering**: Extracts target structure semantic classes (`building`, `wall`, `fence`, `roof`).
    2. **Morphological Filtering**: Performs morphological opening and closing to bridge small gaps and suppress isolated noise pixels.
    3. **Contour Extraction**: Identifies closed external topological boundaries via OpenCV.
    4. **Area Threshold Filtering**: Rejects degenerate contours below configurable area thresholds.
    5. **Polygon Regularization & Simplification**: Employs Douglas-Peucker line simplification (`preserve_topology=True`) to convert noisy raster edges into crisp vector lines.
    6. **Topology Validation & Auto-Repair**: Runs Shapely `is_valid` / `make_valid` routines to guarantee valid, self-intersection-free polygon geometries.
    7. **Parcel Attribute Assignment**: Assigns sequential IDs (`P-001`, `P-002`...), pixel area, perimeter, and solidity metrics.
    """)
