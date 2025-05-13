# streamlit_integration_fixed_v2.py
# Integration module for figure extraction functionality in Streamlit

import streamlit as st
import os
import logging
import tempfile
import shutil
import zipfile
import io
from pathlib import Path
import json
from typing import List, Dict, Optional, Any

# Import the fixed figure extractor
try:
    from figure_extractor_fixed import (
        FigureExtractor,
        FigureExtractionResult,
        PubLayNetModel,
    )

    FIGURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FIGURE_EXTRACTOR_AVAILABLE = False
    logging.warning(
        "Figure extractor module not found. Make sure figure_extractor_fixed_v2.py is in the same directory."
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Figure Extraction Tab Functions ---


def extract_figures_from_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.3,
    progress_callback=None,
) -> Optional[Dict]:
    """Extract figures from PDF using the FigureExtractor with progress callback"""
    if not FIGURE_EXTRACTOR_AVAILABLE:
        logger.error("Figure extractor module not available.")
        return None

    try:
        # Create a figure extractor with model path and confidence threshold
        extractor = FigureExtractor(
            output_dir=output_dir,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

        # Process the PDF
        result = extractor.process_pdf(pdf_path)

        # Convert result to dictionary
        return result.to_dict()
    except Exception as e:
        logger.error(f"Error extracting figures: {e}")
        return None


def initialize_figure_extraction_state():
    """Initialize session state variables for figure extraction"""
    if "figure_extraction_result" not in st.session_state:
        st.session_state.figure_extraction_result = None
    if "figure_output_dir" not in st.session_state:
        st.session_state.figure_output_dir = None
    if "selected_figure_type" not in st.session_state:
        st.session_state.selected_figure_type = "All Figures"
    if "extraction_temp_dir" not in st.session_state:
        st.session_state.extraction_temp_dir = None


def create_zip_file(file_paths: List[str]) -> bytes:
    """Create a zip file from a list of file paths"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in file_paths:
            if os.path.exists(file_path):
                zf.write(file_path, os.path.basename(file_path))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def add_figure_extraction_tab():
    """Add a figure extraction tab to the existing Streamlit app"""
    # Initialize state
    initialize_figure_extraction_state()

    # Create the tab content
    st.header("Extract Figures & Model Architectures")

    # File uploader - reuse the same one from the main analysis tab if possible
    col1, col2 = st.columns([3, 1])

    with col1:
        if (
            "pdf_uploader" in st.session_state
            and st.session_state.pdf_uploader is not None
        ):
            uploaded_file = st.session_state.pdf_uploader
            st.info(f"Using already uploaded file: {uploaded_file.name}")
        else:
            uploaded_file = st.file_uploader(
                "Choose a PDF file", type="pdf", key="figure_pdf_uploader"
            )

    # Add model configuration options
    with st.expander("Advanced Options"):
        model_path = st.text_input(
            "PubLayNet Model Path (optional)", help="Leave empty for automatic download"
        )
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,  # Default threshold
            step=0.05,
            help="Minimum confidence score for figure detection",
        )

        show_captions = st.checkbox(
            "Show Figures with Captions",
            value=True,
            help="When enabled, shows figures with their complete captions",
        )

        quality_filter = st.slider(
            "Quality Filter",
            min_value=0,
            max_value=10,
            value=3,
            help="Higher values filter out more potential false positives (0 = no filtering)",
        )

    with col2:
        extract_button = st.button(
            "Extract Figures",
            disabled=(uploaded_file is None),
            use_container_width=True,
        )

    if not FIGURE_EXTRACTOR_AVAILABLE:
        st.warning(
            "The Figure Extractor module is not available. Make sure figure_extractor_fixed.py is properly installed."
        )

    if extract_button and uploaded_file is not None and FIGURE_EXTRACTOR_AVAILABLE:
        st.session_state.figure_extraction_result = None

        # Create progress bar
        progress_bar = st.progress(0, text="Preparing...")

        # Create a temporary directory for all files
        with tempfile.TemporaryDirectory() as temp_dir:
            st.session_state.extraction_temp_dir = temp_dir

            # Save uploaded file to temp directory
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Create output directory for figures
            figure_output_dir = os.path.join(temp_dir, "figures")
            os.makedirs(figure_output_dir, exist_ok=True)

            # Extract figures with model path and confidence threshold
            try:
                progress_bar.progress(10, text="Analyzing PDF layout...")

                extraction_result = extract_figures_from_pdf(
                    temp_pdf_path,
                    output_dir=figure_output_dir,
                    model_path=model_path if model_path else None,
                    confidence_threshold=confidence_threshold,
                )

                if extraction_result:
                    progress_bar.progress(80, text="Processing extracted figures...")

                    # Create a permanent directory for figures
                    figures_dir = "extracted_figures"
                    os.makedirs(figures_dir, exist_ok=True)

                    # Apply quality filter if set
                    filtered_figures = extraction_result.get("figures", [])
                    if quality_filter > 0:
                        # Filter out low-quality detections
                        filtered_figures = [
                            fig
                            for fig in filtered_figures
                            if (fig.get("confidence", 0) >= quality_filter / 10)
                        ]

                    # Copy figures to the permanent directory
                    for figure in filtered_figures:
                        orig_path = figure["image_path"]
                        # Check if we should use the version with caption
                        if (
                            show_captions
                            and orig_path.endswith(".png")
                            and not orig_path.endswith("_with_caption.png")
                        ):
                            caption_path = orig_path.replace(
                                ".png", "_with_caption.png"
                            )
                            if os.path.exists(caption_path):
                                orig_path = caption_path

                        new_filename = os.path.basename(orig_path)
                        new_path = os.path.join(figures_dir, new_filename)

                        # Copy the file if it exists
                        if os.path.exists(orig_path):
                            shutil.copy2(orig_path, new_path)
                            # Update the path
                            figure["image_path"] = new_path

                    # Handle model architectures separately
                    # Since they're already in the figures list, just update the paths
                    model_architectures = []
                    for fig in filtered_figures:
                        if fig.get("figure_type") == "model_architecture":
                            model_architectures.append(fig)

                    # Update the extraction result with filtered figures
                    extraction_result["figures"] = filtered_figures
                    extraction_result["model_architectures"] = model_architectures

                    # Store results in session state
                    st.session_state.figure_extraction_result = extraction_result
                    st.session_state.figure_output_dir = figures_dir

                    progress_bar.progress(100, text="Extraction Complete!")
                    st.success(
                        f"Successfully extracted {len(filtered_figures)} figures!"
                    )
                else:
                    progress_bar.progress(100, text="Extraction Failed.")
                    st.error("Failed to extract figures from PDF.")

            except Exception as e:
                progress_bar.progress(100, text="Extraction Failed.")
                st.error(f"Error during figure extraction: {str(e)}")

    # Display results if available
    if st.session_state.figure_extraction_result:
        result = st.session_state.figure_extraction_result

        # Add figure type selector
        figure_types = [
            "All Figures",
            "Model Architectures",
            "Graphs",
            "Tables",
            "Other Figures",
        ]
        selected_type = st.selectbox(
            "Filter by figure type:",
            figure_types,
            index=figure_types.index(st.session_state.selected_figure_type),
        )
        st.session_state.selected_figure_type = selected_type

        # Determine which figures to show
        if selected_type == "All Figures":
            figures_to_show = result.get("figures", [])
            st.subheader(f"All Figures ({len(figures_to_show)})")
        elif selected_type == "Model Architectures":
            figures_to_show = result.get("model_architectures", [])
            st.subheader(f"Model Architectures ({len(figures_to_show)})")
        else:
            # Filter by figure type
            figures_to_show = [
                fig
                for fig in result.get("figures", [])
                if (
                    (selected_type == "Graphs" and fig.get("figure_type") == "graph")
                    or (selected_type == "Tables" and fig.get("figure_type") == "table")
                    or (
                        selected_type == "Other Figures"
                        and fig.get("figure_type")
                        not in ["model_architecture", "graph", "table"]
                    )
                )
            ]
            st.subheader(f"{selected_type} ({len(figures_to_show)})")

        # Display figures
        if figures_to_show:
            # Add download button for all figures
            if st.session_state.figure_output_dir:
                file_paths = [
                    fig["image_path"]
                    for fig in figures_to_show
                    if os.path.exists(fig["image_path"])
                ]
                if file_paths:
                    zip_data = create_zip_file(file_paths)
                    st.download_button(
                        label=f"Download All {selected_type}",
                        data=zip_data,
                        file_name=f"{selected_type.lower().replace(' ', '_')}.zip",
                        mime="application/zip",
                    )

            # Display in a grid - create rows with 2 columns each
            for i in range(0, len(figures_to_show), 2):
                cols = st.columns(2)

                # Left column
                with cols[0]:
                    if i < len(figures_to_show):
                        fig = figures_to_show[i]
                        image_path = fig["image_path"]

                        if os.path.exists(image_path):
                            # Display the figure
                            st.image(
                                image_path,
                                caption=f"Figure {fig.get('figure_num', i+1)} (Page {fig.get('page_num')})",
                            )

                            with st.expander("Figure Details"):
                                st.write(
                                    f"**Type:** {fig.get('figure_type', 'Unknown')}"
                                )
                                st.write(
                                    f"**Confidence:** {fig.get('confidence', 0):.2f}"
                                )
                                if fig.get("caption"):
                                    st.write(f"**Caption:** {fig['caption']}")

                                # Add individual download button
                                with open(image_path, "rb") as f:
                                    st.download_button(
                                        label=f"Download This Figure",
                                        data=f.read(),
                                        file_name=os.path.basename(image_path),
                                        mime="image/png",
                                        key=f"download_fig_{i}",
                                    )

                                # Show figure-only version if we're displaying the version with caption
                                if show_captions and "_with_caption" in image_path:
                                    figure_only_path = image_path.replace(
                                        "_with_caption", ""
                                    )
                                    if os.path.exists(figure_only_path):
                                        st.write("**Figure without caption:**")
                                        st.image(figure_only_path, width=300)
                                        with open(figure_only_path, "rb") as f:
                                            st.download_button(
                                                label=f"Download Figure Only",
                                                data=f.read(),
                                                file_name=os.path.basename(
                                                    figure_only_path
                                                ),
                                                mime="image/png",
                                                key=f"download_fig_only_{i}",
                                            )

                # Right column
                with cols[1]:
                    if i + 1 < len(figures_to_show):
                        fig = figures_to_show[i + 1]
                        image_path = fig["image_path"]

                        if os.path.exists(image_path):
                            st.image(
                                image_path,
                                caption=f"Figure {fig.get('figure_num', i+2)} (Page {fig.get('page_num')})",
                            )

                            with st.expander("Figure Details"):
                                st.write(
                                    f"**Type:** {fig.get('figure_type', 'Unknown')}"
                                )
                                st.write(
                                    f"**Confidence:** {fig.get('confidence', 0):.2f}"
                                )
                                if fig.get("caption"):
                                    st.write(f"**Caption:** {fig['caption']}")

                                # Add individual download button
                                with open(image_path, "rb") as f:
                                    st.download_button(
                                        label=f"Download This Figure",
                                        data=f.read(),
                                        file_name=os.path.basename(image_path),
                                        mime="image/png",
                                        key=f"download_fig_{i+1}",
                                    )

                                # Show figure-only version if we're displaying the version with caption
                                if show_captions and "_with_caption" in image_path:
                                    figure_only_path = image_path.replace(
                                        "_with_caption", ""
                                    )
                                    if os.path.exists(figure_only_path):
                                        st.write("**Figure without caption:**")
                                        st.image(figure_only_path, width=300)
                                        with open(figure_only_path, "rb") as f:
                                            st.download_button(
                                                label=f"Download Figure Only",
                                                data=f.read(),
                                                file_name=os.path.basename(
                                                    figure_only_path
                                                ),
                                                mime="image/png",
                                                key=f"download_fig_only_{i+1}",
                                            )
        else:
            st.info(f"No figures of type '{selected_type}' found.")
    else:
        st.info(
            "Upload a PDF and click 'Extract Figures' to detect and visualize figures in the paper."
        )


def integrate_with_main_app(tab_container):
    """Integrate figure extraction with main app by adding it as a tab"""
    add_figure_extraction_tab()
