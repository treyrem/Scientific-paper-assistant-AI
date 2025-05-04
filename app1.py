import streamlit as st
import tempfile
import os
import sys
import pandas as pd
import altair as alt


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from layout_analyzer import analyze_paper

st.title("Scientific Paper Layout Analyzer")

uploaded_file = st.file_uploader("Upload a PDF of a scientific paper", type=["pdf"])
if not uploaded_file:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    pdf_path = tmp.name

try:
    with st.spinner("Analyzing document..."):
        doc_info, all_fragments, all_blocks, by_type, fragment_summary = analyze_paper(pdf_path)
    
    st.write(f"Document has **{doc_info['total_pages']}** pages.")
    st.write(f"Extracted **{fragment_summary['total_fragments']}** total text fragments.")
    
    tab1, tab2, tab3 = st.tabs(["Fragment Summary", "Layout Blocks", "Text Analysis"])
    
    with tab1:
        st.header("Fragment Summary")
        
        st.subheader("Fragments per Page")
        page_data = pd.DataFrame({
            'Page': list(fragment_summary['fragments_per_page'].keys()),
            'Count': list(fragment_summary['fragments_per_page'].values())
        })
        
        if not page_data.empty:
            chart = alt.Chart(page_data).mark_bar().encode(
                x='Page:O',
                y='Count:Q',
                tooltip=['Page', 'Count']
            ).properties(width=600)
            st.altair_chart(chart)
        
        st.subheader("Fragments by Type")
        type_data = pd.DataFrame({
            'Type': list(fragment_summary['fragments_by_type'].keys()),
            'Count': list(fragment_summary['fragments_by_type'].values())
        })
        
        if not type_data.empty:
            chart = alt.Chart(type_data).mark_bar().encode(
                x='Type:N',
                y='Count:Q',
                color='Type:N',
                tooltip=['Type', 'Count']
            ).properties(width=600)
            st.altair_chart(chart)
        
       
        st.subheader("Fragment Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Chars per Fragment", f"{fragment_summary['avg_chars_per_fragment']:.1f}")
        with col2:
            st.metric("Total Word Count", fragment_summary['word_count'])
        with col3:
            st.metric("Fragments with Numbers", fragment_summary['fragments_with_numbers'])
        
        # Most common words
        st.subheader("Most Common Words")
        if fragment_summary['most_common_words']:
            word_data = pd.DataFrame(fragment_summary['most_common_words'], columns=['Word', 'Frequency'])
            chart = alt.Chart(word_data).mark_bar().encode(
                x='Frequency:Q',
                y=alt.Y('Word:N', sort='-x'),
                tooltip=['Word', 'Frequency']
            ).properties(height=300)
            st.altair_chart(chart)
    
    with tab2:
        st.header("Layout Blocks")
        
        for btype, blocks in by_type.items():
            with st.expander(f"{btype} ({len(blocks)})", expanded=False):
                for i, blk in enumerate(blocks):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**Page {blk['page']}** | Block {i+1}")
                        if blk["crop"] and os.path.exists(blk["crop"]):
                            try:
                                st.image(blk["crop"], width=200)
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")
                    with col2:
                        # Show fragment count instead of full text
                        fragment_lines = blk["text"].strip().split("\n")
                        total_chars = len(blk["text"])
                        st.write(f"**Fragment Count:** {len(fragment_lines)}")
                        st.write(f"**Characters:** {total_chars}")
                        st.write(f"**First few words:** {' '.join(blk['text'].split()[:10])}...")
                    st.divider()
    
    with tab3:
        st.header("Text Content Analysis")
        
        # Text statistics
        st.subheader("Text Statistics")
        
        # Create a dataframe of word frequencies
        word_freq_df = pd.DataFrame(fragment_summary['word_frequency'].most_common(100), 
                                    columns=['Word', 'Frequency'])
        
        # Filter options
        min_length = st.slider("Minimum word length", 1, 10, 3)
        filtered_words = word_freq_df[word_freq_df['Word'].str.len() >= min_length]
        
        # Show word cloud or bar chart based on frequency
        st.write(f"Top words (minimum length: {min_length})")
        chart = alt.Chart(filtered_words.head(20)).mark_bar().encode(
            x='Frequency:Q',
            y=alt.Y('Word:N', sort='-x'),
            tooltip=['Word', 'Frequency']
        ).properties(height=400)
        st.altair_chart(chart)
        
        # Fragment length distribution
        if all_fragments:
            fragment_lengths = [len(f["text"]) for f in all_fragments]
            length_df = pd.DataFrame({'Length': fragment_lengths})
            st.subheader("Fragment Length Distribution")
            chart = alt.Chart(length_df).mark_bar().encode(
                x=alt.X('Length:Q', bin=True),
                y='count()'
            ).properties(width=600)
            st.altair_chart(chart)
            
except Exception as e:
    st.error(f"Error analyzing document: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# Clean up temporary file
finally:
    try:
        os.unlink(pdf_path)
    except:
        pass