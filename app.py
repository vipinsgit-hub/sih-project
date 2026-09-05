"""
SIH Cadastral AI Prototype
Smart India Hackathon: AI-Based Automated Urban Parcel Mapping & Cadastral Feature Extraction
"""
import streamlit as st
from src import __version__

# Page configuration
st.set_page_config(
    page_title="CadastralAI - Urban Parcel Mapping Prototype",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background: #DCFCE7;
        color: #166534;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/map-marker.png", width=64)
    st.title("CadastralAI")
    st.caption(f"Prototype Core Engine v{__version__}")
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    st.info("Milestone 1: Project Architecture & Environment Initialized")
    
    st.markdown("### ⚙️ System Status")
    st.success("✔ Environment: Online")
    st.success("✔ Modular Pipeline: Ready")

# Main Content
st.markdown('<div class="main-header">AI-Based Automated Urban Parcel Mapping</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cadastral Feature Extraction & Encroachment Detection Using Drone Imagery</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="status-card">
        <h4>🚀 Prototype Architecture Initialized</h4>
        <p>This technical proof-of-concept demonstrates the feasibility of an end-to-end AI workflow for urban parcel extraction, boundary regularization, historical cadastral verification, and surveyor validation.</p>
        <span class="badge">Milestone 1 Complete</span>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Workflow Stages")
    steps = [
        ("1. Aerial Drone Image Preprocessing", "Contrast enhancement, tiling, orthorectification prep"),
        ("2. AI Feature & Boundary Segmentation", "Deep learning edge/boundary wall detection"),
        ("3. Parcel Vectorization & Regularization", "Contour simplification & topology enforcement"),
        ("4. GIS Representation & Overlay", "Interactive multi-layer geospatial visualization"),
        ("5. Historical Cadastral Change Detection", "IoU overlap & encroachment discrepancy alerts"),
        ("6. Surveyor Verification & GeoJSON Export", "Manual correction interface & standard GIS export"),
    ]
    for title, desc in steps:
        with st.expander(title, expanded=False):
            st.write(desc)

with col2:
    st.subheader("System Information")
    st.metric(label="Module Version", value=f"v{__version__}")
    st.metric(label="Pipeline Status", value="Ready for M2")
    st.metric(label="Modules Loaded", value="5 / 5")
    
    st.markdown("---")
    st.caption("Smart India Hackathon Prototype | Developed by Team CadastralAI")
