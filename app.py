"""
SIH Cadastral AI Prototype
Milestone 3: AI Feature & Semantic Segmentation Pipeline
"""
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

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CadastralAI - Drone Parcel Mapping & Segmentation",
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
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/map-marker.png", width=56)
    st.title("CadastralAI")
    st.caption(f"Prototype Engine v{__version__}")
    st.markdown("---")

    st.markdown("### 🛠️ Active Milestone")
    st.markdown("**Milestone 3**: AI Feature & Semantic Segmentation")
    
    st.markdown("---")
    st.markdown("### 🧠 Model Configuration")
    model_choice = st.selectbox(
        "Semantic Segmentation Model",
        options=[DEFAULT_MODEL_ID],
        index=0,
        help="Lightweight Transformer model pre-trained for semantic scene parsing"
    )
    
    overlay_alpha = st.slider(
        "Overlay Transparency (Alpha)",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Blending weight between original drone photo and AI segmentation map"
    )

    st.markdown("---")
    st.caption("Smart India Hackathon Prototype | Team CadastralAI")

# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">AI Cadastral Mapping</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Urban Parcel Feature Extraction & Semantic Segmentation</div>', unsafe_allow_html=True)

# Disclaimer Box
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Technical Distinction & Prototype Notice:</strong>
    The AI segmentation stage identifies physical visual elements (buildings, roads, walls, vegetation, ground).
    It extracts candidate geometry for subsequent parcel regularizers and does <strong>not</strong> independently create legally binding cadastral land ownership records.
</div>
""", unsafe_allow_html=True)

# Main Workspace Tabs
tab_analysis, tab_workflow = st.tabs(["🚀 Cadastral AI Analysis", "ℹ️ Architecture & Methodology"])

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

            # Run Segmentation Button
            st.subheader("2. AI Semantic Feature Segmentation")
            col_btn, col_info = st.columns([1, 2])
            with col_btn:
                run_ai = st.button("⚡ Run AI Feature Segmentation", type="primary", use_container_width=True)

            # Execution logic
            if run_ai or "seg_result" in st.session_state:
                if run_ai:
                    with st.spinner("🧠 Loading AI SegFormer model & computing semantic feature map..."):
                        model_bundle = get_cached_model(model_choice)
                        seg_result = segment_image(loaded_image, model_bundle)
                        st.session_state["seg_result"] = seg_result
                else:
                    seg_result = st.session_state["seg_result"]

                # Display Visual Results
                st.markdown("### 🗺️ Visual Segmentation Diagnostics")
                
                v_tab1, v_tab2, v_tab3 = st.tabs(["Overlay Blended Map", "Color-Coded Semantic Mask", "Original Survey Photo"])

                with v_tab1:
                    # Dynamically recompute overlay based on current alpha slider
                    dynamic_overlay = create_segmentation_overlay(
                        loaded_image,
                        seg_result["mask"],
                        alpha=overlay_alpha,
                        palette=model_bundle.get("palette") if "model_bundle" in locals() else None
                    )
                    st.image(dynamic_overlay, caption=f"AI Semantic Segmentation Overlay (Alpha: {overlay_alpha:.2f})", use_container_width=True)

                with v_tab2:
                    st.image(seg_result["colored_mask"], caption="Semantic Class Mask (Color-Coded by ADE20K Feature Categories)", use_container_width=True)

                with v_tab3:
                    st.image(loaded_image, caption=f"Original Input: {metadata['filename']}", use_container_width=True)

                # Detected Features Section
                st.markdown("### 🏷️ Identified Land Features")
                
                badges_html = "".join([f'<span class="badge-primary">{c}</span>' for c in seg_result["detected_classes"]])
                st.markdown(badges_html, unsafe_allow_html=True)

                # Class Statistics Table
                st.markdown("### 📊 Semantic Feature Statistics")
                df_stats = pd.DataFrame(seg_result["class_statistics"])
                if not df_stats.empty:
                    df_stats.rename(columns={
                        "class_name": "Detected Semantic Feature",
                        "class_id": "Class ID",
                        "pixel_count": "Pixel Count",
                        "percentage": "Area Coverage (%)"
                    }, inplace=True)
                    st.dataframe(df_stats[["Detected Semantic Feature", "Class ID", "Pixel Count", "Area Coverage (%)"]], use_container_width=True, hide_index=True)
                
                st.success("✔ AI Semantic Segmentation Completed successfully. Ready for Milestone 4 (Boundary Extraction & Polygon Regularization).")

        except ImageProcessingError as e:
            st.error(f"⚠️ Image Error: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ Pipeline Error: {str(e)}")
    else:
        st.info("👆 Upload an aerial survey image or click 'Load Demo Aerial Survey' above to begin.")

with tab_workflow:
    st.markdown("""
    ### 🔬 AI Segmentation Model Details
    - **Architecture**: `SegFormer-B0` (Hierarchical Transformer Encoder + Lightweight MLP Decoder)
    - **Pretrained Dataset**: ADE20K (150 semantic categories including buildings, roads, vegetation, walls, fences, terrain)
    - **Inference Hardware**: Dynamic (CUDA GPU if available, optimized multi-threaded CPU fallback)
    - **Resolution**: Native auto-upsampled to source drone imagery dimensions for pixel-level boundary fidelity.
    
    ### 🛡️ Cadastral Engineering Principle
    Visual feature segmentation extracts candidate obstacle edges, building footprints, and roads. In subsequent milestones, geometric regularization algorithms will convert these raw visual masks into closed, topology-valid parcel polygons for surveyor verification.
    """)
