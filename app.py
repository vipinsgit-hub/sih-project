"""
SIH Cadastral AI Prototype
Milestone 2: Aerial Image Input & Preprocessing Pipeline
"""
import os
import streamlit as st
from PIL import Image

from src import __version__
from src.utils.image_processing import (
    load_image,
    get_image_metadata,
    preprocess_for_model,
    ImageProcessingError,
    SUPPORTED_FORMATS,
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CadastralAI - Drone Parcel Mapping Prototype",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
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
        margin-bottom: 1.5rem;
    }
    .info-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
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
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        background: #DCFCE7;
        color: #166534;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/map-marker.png", width=56)
    st.title("CadastralAI")
    st.caption(f"Prototype Engine v{__version__}")
    st.markdown("---")

    st.markdown("### 🛠️ Active Milestone")
    st.markdown("**Milestone 2**: Aerial Image Input & Preprocessing")
    
    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Configuration")
    target_dim = st.selectbox(
        "AI Input Target Size",
        options=[512, 640, 768, 1024],
        index=0,
        help="Target square dimension for AI model inference tensor"
    )
    normalize_option = st.checkbox("ImageNet Standardization (Mean/Std)", value=True)

    st.markdown("---")
    st.caption("Smart India Hackathon Prototype | Team CadastralAI")

# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">AI Cadastral Mapping</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Drone-Based Urban Parcel Analysis & Feature Extraction</div>', unsafe_allow_html=True)

# Tabs / Workflow Container
tab_upload, tab_pipeline_info = st.tabs(["📸 Aerial Image Input & Preprocessing", "ℹ️ Architecture & Workflow"])

with tab_upload:
    st.subheader("1. Upload Drone / Aerial Survey Imagery")
    
    col_input, col_sample = st.columns([3, 1])
    with col_input:
        uploaded_file = st.file_uploader(
            "Choose an aerial image (JPG, PNG, TIFF)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="High-resolution aerial or drone survey image of urban parcels"
        )
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button("📁 Load Demo Aerial Image", use_container_width=True)

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
            # 1. Safe loading
            loaded_image = load_image(image_bytes)
            
            # 2. Metadata extraction
            metadata = get_image_metadata(
                image=loaded_image,
                filename=filename,
                file_size_bytes=file_size
            )

            # 3. AI Preprocessing
            preprocessed = preprocess_for_model(
                image=loaded_image,
                target_size=(target_dim, target_dim),
                normalize_imagenet=normalize_option
            )

            st.markdown("---")

            # Display side-by-side: Original Preview & Processed Diagnostics
            col_img, col_meta = st.columns([3, 2])

            with col_img:
                st.subheader("Original Aerial Image")
                st.image(
                    loaded_image,
                    caption=f"Uploaded: {metadata['filename']} ({metadata['resolution']})",
                    use_container_width=True
                )

            with col_meta:
                st.subheader("Image Information")
                
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Resolution</div>
                        <div class="metric-value">{metadata['resolution']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Color Channels</div>
                        <div class="metric-value">{metadata['channels']} ({', '.join(metadata['channel_names'])})</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Format</div>
                        <div class="metric-value">{metadata['format']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">File Size</div>
                        <div class="metric-value">{metadata.get('file_size_formatted', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.subheader("Preprocessing Status")
                st.success("✔ Image safely validated and loaded")
                st.success("✔ Converted to standardized RGB representation")
                st.success(f"✔ Resized AI-tensor: ({preprocessed['normalized_tensor_np'].shape[0]}, {target_dim}, {target_dim})")
                st.success(f"✔ Model input normalized: (Scale X: {preprocessed['scale_factors'][0]:.2f}, Y: {preprocessed['scale_factors'][1]:.2f})")
                
                st.info("Ready for Milestone 3: AI Feature & Boundary Segmentation Inference")

        except ImageProcessingError as e:
            st.error(f"⚠️ Image Processing Error: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error while processing image: {str(e)}")
    else:
        st.info("👆 Upload an aerial survey image or click 'Load Demo Aerial Image' to begin.")

with tab_pipeline_info:
    st.markdown("""
    ### 🔄 Technical Workflow & Principles
    
    1. **Preservation of Raw Imagery**: The original uploaded aerial image is strictly maintained intact at full native resolution for human surveyor inspection.
    2. **AI-Ready Standardization**: A detached copy is normalized into standard PyTorch tensor format `(C, H, W)` with ImageNet statistical normalization.
    3. **Coordinate Scale Mapping**: Scale factors `(scale_x, scale_y)` are computed dynamically to enable accurate backward-projection of AI parcel coordinates to real-world geospatial dimensions.
    """)
