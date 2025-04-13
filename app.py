import streamlit as st
import pandas as pd
import time
from datetime import datetime
import plotly.express as px

from sentiment_analyzer import (
    perform_basic_sentiment_analysis,
    perform_emotion_analysis,
    perform_aspect_based_analysis
)
from visualization import (
    create_sentiment_gauge,
    create_emotion_bar_chart,
    create_aspect_sentiment_chart
)
from utils import get_sample_texts

# Configure the page
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

def analyze_text(text):
    """Analyze the given text and return the results"""
    # Basic sentiment analysis
    sentiment_result = perform_basic_sentiment_analysis(text)
    
    # Emotion analysis
    emotion_result = perform_emotion_analysis(text)
    
    # Aspect based sentiment analysis
    aspect_result = perform_aspect_based_analysis(text)
    
    # Add to history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis_result = {
        "timestamp": timestamp,
        "text": text,
        "sentiment": sentiment_result,
        "emotions": emotion_result,
        "aspects": aspect_result
    }
    
    st.session_state.history.append(analysis_result)
    
    return analysis_result

def process_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("The CSV file must contain a 'text' column.")
            return None
        
        results = []
        progress_bar = st.progress(0)
        
        for i, row in enumerate(df.itertuples()):
            text = getattr(row, 'text')
            result = analyze_text(text)
            results.append(result)
            progress_bar.progress((i + 1) / len(df))
        
        return pd.DataFrame(results)
    
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None

# Header section
st.title("Sentiment Analysis Dashboard")
st.markdown("Analyze the emotional content of your text using advanced NLP techniques")

# Main content area
tab1, tab2, tab3 = st.tabs(["Text Analysis", "Batch Analysis", "History"])

with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input section
        st.subheader("Enter Text to Analyze")
        
        # Sample text selection
        sample_option = st.selectbox(
            "Try a sample text or enter your own below:",
            ["Custom text..."] + get_sample_texts()
        )
        
        if sample_option == "Custom text...":
            text_input = st.text_area(
                "Enter text for analysis:",
                height=150,
                placeholder="Type or paste your text here for sentiment analysis..."
            )
        else:
            text_input = sample_option
            st.text_area("Sample text:", value=text_input, height=150, disabled=True)
    
    with col2:
        st.subheader("Options")
        analyze_aspects = st.checkbox("Analyze aspects/topics", value=True)
        detect_emotions = st.checkbox("Detect emotions", value=True)
    
    # Analysis button
    if st.button("Analyze Sentiment", type="primary"):
        if not text_input:
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                result = analyze_text(text_input)
                
                # Display results in expandable sections
                st.subheader("Analysis Results")
                
                # Create columns for displaying results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Overall sentiment
                    st.markdown("### Overall Sentiment")
                    create_sentiment_gauge(result["sentiment"]["compound"])
                    st.markdown(f"**Classification**: {result['sentiment']['classification']}")
                    st.markdown(f"**Confidence**: {result['sentiment']['confidence']:.2f}")
                
                with col2:
                    # Emotion analysis
                    if detect_emotions:
                        st.markdown("### Detected Emotions")
                        create_emotion_bar_chart(result["emotions"])
                
                # Aspect-based analysis
                if analyze_aspects and result["aspects"]:
                    st.markdown("### Aspect-Based Sentiment")
                    create_aspect_sentiment_chart(result["aspects"])
                    
                    # Display aspects in a table
                    aspects_df = pd.DataFrame(result["aspects"])
                    st.dataframe(aspects_df)

with tab2:
    st.subheader("Batch Analysis from CSV")
    st.markdown("""
    Upload a CSV file with a column named 'text' containing the text to analyze.
    Results will be available for download after processing.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("Process CSV", type="primary"):
            with st.spinner("Processing CSV file..."):
                results_df = process_csv(uploaded_file)
                
                if results_df is not None:
                    st.success(f"Successfully analyzed {len(results_df)} entries")
                    
                    # Display summary
                    st.subheader("Summary")
                    
                    # Create sentiment distribution chart
                    sentiment_counts = results_df['sentiment'].apply(
                        lambda x: x['classification']
                    ).value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    
                    fig = px.pie(
                        sentiment_counts, 
                        values='Count', 
                        names='Sentiment',
                        color='Sentiment',
                        color_discrete_map={
                            'Positive': '#38B2AC', 
                            'Neutral': '#718096',
                            'Negative': '#FF6B6B'
                        },
                        title='Sentiment Distribution'
                    )
                    st.plotly_chart(fig)
                    
                    # Prepare CSV for download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )

with tab3:
    st.subheader("Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history yet. Analyze some text to see results here.")
    else:
        # Create a dataframe from history
        history_data = []
        for item in st.session_state.history:
            history_data.append({
                "Timestamp": item["timestamp"],
                "Text": item["text"][:50] + "..." if len(item["text"]) > 50 else item["text"],
                "Sentiment": item["sentiment"]["classification"],
                "Score": item["sentiment"]["compound"]
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Visualization of history
        if len(history_df) > 1:
            st.subheader("Sentiment Trend")
            
            fig = px.line(
                history_df, 
                x="Timestamp", 
                y="Score",
                color_discrete_sequence=["#5B61F9"],
                markers=True,
                title="Sentiment Score Over Time"
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Sentiment Score (-1 to 1)",
                yaxis_range=[-1, 1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# Footer with information
st.markdown("---")
st.markdown("""
This dashboard uses NLP techniques to analyze sentiment and emotions in text.
- **Sentiment Score**: Ranges from -1 (very negative) to 1 (very positive)
- **Emotions**: Detects joy, sadness, anger, fear, surprise, and more
- **Aspect Analysis**: Extracts topics and their associated sentiments
""")
