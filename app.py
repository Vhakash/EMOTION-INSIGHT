import streamlit as st
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
from database import (
    save_analysis,
    get_all_analyses,
    get_analysis_by_id,
    delete_analysis,
    delete_all_analyses,
    get_sentiment_distribution,
    get_sentiment_history_dataframe
)

# Helper function for Plotly charts to prevent duplicate keys
def safe_plotly_chart(fig, container_width=True, key=None):
    """
    Display a Plotly chart with a guaranteed unique key to prevent duplicate element IDs.
    
    Parameters:
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure to display
    container_width : bool, default=True
        If True, the chart will use the full width of the container
    key : str, optional
        A unique key for the chart element. If not provided, a random key will be generated.
    """
    import random
    import string
    
    if key is None:
        # Generate a random string if no key is provided
        key = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        
    st.plotly_chart(fig, use_container_width=container_width, key=key)

# Configure the page
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
    
# Load history from database when the app starts
if 'db_initialized' not in st.session_state:
    try:
        st.session_state.history = get_all_analyses()
        st.session_state.db_initialized = True
    except Exception as e:
        st.error(f"Error loading data from database: {str(e)}")
        st.session_state.db_initialized = False

def analyze_text(text):
    """Analyze the given text and return the results"""
    # Basic sentiment analysis
    sentiment_result = perform_basic_sentiment_analysis(text)
    
    # Emotion analysis
    emotion_result = perform_emotion_analysis(text)
    
    # Aspect based sentiment analysis
    aspect_result = perform_aspect_based_analysis(text)
    
    # Create result dictionary
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis_result = {
        "timestamp": timestamp,
        "text": text,
        "sentiment": sentiment_result,
        "emotions": emotion_result,
        "aspects": aspect_result
    }
    
    # Save to database
    try:
        analysis_id = save_analysis(analysis_result)
        # If successful, reload all analyses from the database
        st.session_state.history = get_all_analyses()
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        # Add to session state history as fallback
        st.session_state.history.append(analysis_result)
    
    return analysis_result

def process_csv(uploaded_file):
    try:
        # Try different encoding options and error handling
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        except:
            try:
                df = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
            except:
                df = pd.read_csv(uploaded_file, encoding='cp1252', on_bad_lines='skip')
        
        if 'text' not in df.columns:
            st.error("The CSV file must contain a 'text' column.")
            return None
        
        # Clean the text column
        df['text'] = df['text'].astype(str).apply(lambda x: x.strip())
        df = df[df['text'].str.len() > 0]  # Remove empty rows
        
        results = []
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for i, row in enumerate(df.itertuples()):
            try:
                text = getattr(row, 'text')
                result = analyze_text(text)
                results.append(result)
                progress_bar.progress((i + 1) / total_rows)
            except Exception as e:
                st.warning(f"Skipped row {i+1} due to error: {str(e)}")
                continue
        
        if not results:
            st.error("No valid entries were processed.")
            return None
            
        return pd.DataFrame(results)
    
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None
st.markdown("""
<div class="title-container">
    <h1>✨ Emotion Insight Dashboard ✨</h1>
    <p class="subtitle">Discover the emotional landscape of your text with advanced AI analysis</p>
    <div style="margin-top: 20px;">
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😊</span>
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😢</span>
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😡</span>
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😲</span>
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😍</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Text Analysis 📝", 
    "Batch Analysis 📊",
    "History 📜",
    "Analytics 📈"
])
# Add these imports for enhanced visuals
import random
from PIL import Image
from io import BytesIO
import base64

# Enhanced CSS with more animations and visual effects
st.markdown("""
<style>
    /* Existing animations remain the same ... */
    
    /* Enhanced title container */
    .title-container {
        background: #4A90E2;  /* Solid background color */
        padding: 3rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        border: 4px solid #FFD700;
        position: relative;
        z-index: 1;
    }
    
    .title-container h1 {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        text-shadow: 4px 4px 8px rgba(0, 0, 0, 0.4);
        color: #FFD700;
        font-weight: 900;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    
    .subtitle {
        color: #FFFFFF;
        font-size: 1.5rem;
        margin-top: 1rem;
        font-weight: 500;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    background: linear-gradient(-45deg, #FF6B6B, #4ECDC4, #45B7D1, #9B6BFF);
    background-size: 400% 400%;
    animation: colorShift 15s ease infinite;
    padding: 2.5rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    color: #FFE5D9;  /* Changed from #E2F3FF to warm peach */
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    .title-container h1 {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        background: linear-gradient(90deg, #FFF200, #FFB800);  /* Brighter yellow gradient */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        background: linear-gradient(90deg, #7FFFD4, #00FFFF);  /* Aquamarine to cyan */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Enhanced cards */
    .card {
        border-radius: 12px;
        padding: 1.8rem;
        background: linear-gradient(135deg, #2C3E50, #3498DB);
        color: #FFE5D9;  /* Changed from #E2F3FF to warm peach */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        animation: slideIn 1s ease-out;
        transition: all 0.4s ease;
        border-top: 5px solid transparent;
        border-image: linear-gradient(to right, #FF6B6B, #4ECDC4);
        border-image-slice: 1;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s ease;
        font-weight: 600;
        border-radius: 8px;
        padding: 5px 10px;
        background: linear-gradient(90deg, rgba(52, 152, 219, 0.2), rgba(41, 128, 185, 0.2));
        color: #FFE5D9 !important;  /* Changed from #E2F3FF to warm peach */
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: #FFE5D9 !important;  /* Changed from #E2F3FF to warm peach */
    }
    
    /* Button enhancements */
    .stButton button {
        transition: all 0.3s ease;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: #FFE5D9;  /* Changed from white to warm peach */
    }
</style>
""", unsafe_allow_html=True)

# Helper function to create animated loading
def show_animated_loading(message="Analyzing..."):
    with st.spinner(message):
        loading_html = f"""
        <div class="loading-animation">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <h3 style="margin-left: 15px;">{message}</h3>
        </div>
        """
        loading_placeholder = st.empty()
        loading_placeholder.markdown(loading_html, unsafe_allow_html=True)
        return loading_placeholder

# Helper function to create tooltip
def tooltip(text, tooltip_text):
    return f"""
    <div class="tooltip">{text}
        <span class="tooltiptext">{tooltip_text}</span>
    </div>
    """

# Helper function to create sentiment badge
def sentiment_badge(sentiment):
    colors = {
        "Positive": "#38B2AC",
        "Negative": "#FF6B6B",
        "Neutral": "#718096"
    }
    icons = {
        "Positive": "😊",
        "Negative": "😞",
        "Neutral": "😐"
    }
    color = colors.get(sentiment, "#718096")
    icon = icons.get(sentiment, "❓")
    
    return f"""
    <div style="display: inline-block; padding: 5px 12px; background-color: {color}; 
                color: white; border-radius: 20px; font-weight: bold; margin: 5px 0;">
        {icon} {sentiment}
    </div>
    """

# Enhanced header with animated background
st.markdown("""
<div class="title-container">
    <h1>✨ Emotion Insight Dashboard ✨</h1>
    <p class="subtitle">Discover the emotional landscape of your text with advanced AI analysis</p>
    <div style="margin-top: 20px;">
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😊</span>
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😢</span>
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😡</span>
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😲</span>
        <span style="font-size: 2rem; margin: 0 10px;" class="float">😍</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ... existing tab creation code ...

with tab1:
    # Enhanced intro card with interactive elements
    st.markdown("""
    <div class="card">
        <h3><span class="emoji-icon">💡</span> How it works</h3>
        <p>Our AI-powered sentiment analysis engine examines your text to detect emotions, sentiment, and key topics.
        The analysis uses advanced natural language processing to understand context and nuance.</p>
        <div style="display: flex; justify-content: space-around; margin-top: 20px; text-align: center;">
            <div>
                <div style="font-size: 2rem; margin-bottom: 10px;">📝</div>
                <p><strong>Input Text</strong></p>
            </div>
            <div style="font-size: 1.5rem; margin-top: 15px;">➡️</div>
            <div>
                <div style="font-size: 2rem; margin-bottom: 10px;">🧠</div>
                <p><strong>AI Analysis</strong></p>
            </div>
            <div style="font-size: 1.5rem; margin-top: 15px;">➡️</div>
            <div>
                <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
                <p><strong>Insights</strong></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Text input section with animated card
        st.markdown("""
        <div class="card">
            <h3>Enter Text to Analyze</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample text selection with enhanced UI
        sample_option = st.selectbox(
            "Try a sample text or enter your own below:",
            ["Custom text..."] + get_sample_texts()
        )
        
        if sample_option == "Custom text...":
            text_input = st.text_area(
                "Enter text for analysis:",
                height=150,
                placeholder="Type or paste your text here for sentiment analysis...",
                help="For best results, use texts with at least a few sentences to provide enough context for the analysis."
            )
        else:
            text_input = sample_option
            st.text_area("Sample text:", value=text_input, height=150, disabled=True)
    
    with col2:
        # Options in an animated card
        st.markdown("""
        <div class="card">
            <h3><span class="emoji-icon">⚙️</span> Analysis Options</h3>
        </div>
        """, unsafe_allow_html=True)
        
        analyze_aspects = st.checkbox(
            "Analyze aspects/topics", 
            value=True,
            help="Extract specific topics or aspects mentioned in the text and analyze sentiment for each"
        )
        
        detect_emotions = st.checkbox(
            "Detect emotions", 
            value=True,
            help="Identify specific emotions expressed in the text beyond positive/negative sentiment"
        )
        
        # Add some extra interactive options
        st.markdown("#### Text Complexity")
        complexity = st.slider(
            "Analysis Depth", 
            min_value=1, 
            max_value=3, 
            value=2,
            help="Higher depth provides more detailed analysis but may take longer"
        )
    
    # Create a centered container for the button
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        # Analysis button with animation
        st.markdown('<div style="text-align: center; padding: 10px 0;">', unsafe_allow_html=True)
        analyze_button = st.button("✨ Analyze Sentiment ✨", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if analyze_button:
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
                    create_sentiment_gauge(result["sentiment"]["compound"], key="result_main")
                    st.markdown(f"**Classification**: {result['sentiment']['classification']}")
                    st.markdown(f"**Confidence**: {result['sentiment']['confidence']:.2f}")
                
                with col2:
                    # Emotion analysis
                    if detect_emotions:
                        st.markdown("### Detected Emotions")
                        create_emotion_bar_chart(result["emotions"], key="result_emotions")
                        
                        # Get dominant emotion
                        dominant_emotion = max(result["emotions"].items(), key=lambda x: x[1])
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 10px;">
                            <p>Dominant emotion: <strong>{dominant_emotion[0]}</strong> ({dominant_emotion[1]:.2f})</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Aspect-based analysis
                if analyze_aspects and result["aspects"]:
                    st.markdown("### Aspect-Based Sentiment")
                    create_aspect_sentiment_chart(result["aspects"], key="result_aspects")
                    
                    # Display aspects in a table
                    aspects_df = pd.DataFrame(result["aspects"])
                    st.dataframe(aspects_df)

with tab2:
    # Batch Analysis with animated card
    st.markdown("""
    <div class="card">
        <h3><span class="emoji-icon">📊</span> Batch Analysis from CSV</h3>
        <p>Upload a CSV file with a column named <code>text</code> containing the text to analyze.
        Results will be available for download after processing.</p>
        <p>Perfect for analyzing customer reviews, survey responses, or social media comments in bulk!</p>
    </div>
    """, unsafe_allow_html=True)
    
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
                    st.plotly_chart(fig, key="csv_sentiment_pie")
                    
                    # Prepare CSV for download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )

with tab3:
    # History tab with animated card
    st.markdown("""
    <div class="card">
        <h3><span class="emoji-icon">📜</span> Analysis History</h3>
        <p>View and manage your past analyses. Select any entry to see detailed results or export your analysis history.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.history:
        # Empty state with friendly message
        st.markdown("""
        <div style="text-align: center; padding: 40px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;">
            <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
            <h3>No Analysis History Yet</h3>
            <p>After you analyze some text, your results will appear here for future reference.</p>
            <p>Try analyzing text in the Text Analysis tab to get started!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create a dataframe from history
        history_data = []
        for item in st.session_state.history:
            # Handle different data formats (database vs direct analysis)
            if "sentiment" in item:
                # Direct analysis format
                sentiment = item["sentiment"]["classification"]
                score = item["sentiment"]["compound"]
            else:
                # Database format
                sentiment = item["sentiment_classification"]
                score = item["sentiment_score"]
                
            history_data.append({
                "ID": item.get("id"),
                "Timestamp": item["timestamp"],
                "Text": item["text"][:50] + "..." if len(item["text"]) > 50 else item["text"],
                "Sentiment": sentiment,
                "Score": score
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Add management tools
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create a dictionary to map IDs to display text for the format_func
            id_to_display = {row['ID']: f"{row['Timestamp']} - {row['Text']}" for index, row in history_df.iterrows()}
            
            selected_id = st.selectbox("Select an entry to view details:", 
                                          options=history_df['ID'],
                                          format_func=lambda id: id_to_display[id])
        
        with col2:
            if st.button("Delete Selected Entry", key="delete_entry"):
                try:
                    if delete_analysis(selected_id):
                        st.success(f"Successfully deleted analysis #{selected_id}")
                        # Reload history from database and rerun the script
                        st.session_state.history = get_all_analyses()
                        st.rerun()
                    else:
                        st.warning(f"Entry #{selected_id} not found in database.")
                except Exception as e:
                    st.error(f"Error deleting entry: {str(e)}")
        
        # Display the dataframe
        st.dataframe(history_df, use_container_width=True)
        
        # Show details of selected entry
        if st.session_state.history:
            # Find the selected item from session_state using the selected_id
            selected_item = next((item for item in st.session_state.history if item.get("id") == selected_id), None)
            
            if selected_item:
                st.subheader(f"Details for Analysis from {selected_item['timestamp']}")
                
                # Display text
                st.markdown("### Text")
                st.text_area("Analyzed Text", value=selected_item["text"], height=100, disabled=True)
                
                # Create expandable sections for results
                with st.expander("Sentiment Analysis Results", expanded=True):
                    # Create columns for displaying detailed results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Overall sentiment
                        st.markdown("#### Overall Sentiment")
                        # Handle different formats
                        if "sentiment" in selected_item:
                            # Direct analysis format
                            create_sentiment_gauge(selected_item["sentiment"]["compound"], key=f"history_{selected_id}_sentiment")
                            st.markdown(f"**Classification**: {selected_item['sentiment']['classification']}")
                            st.markdown(f"**Confidence**: {selected_item['sentiment']['confidence']:.2f}")
                        else:
                            # Database format
                            create_sentiment_gauge(selected_item["sentiment_score"], key=f"history_{selected_id}_sentiment")
                            st.markdown(f"**Classification**: {selected_item['sentiment_classification']}")
                            st.markdown(f"**Confidence**: {selected_item['confidence']:.2f}")
                    
                    with col2:
                        # Emotion analysis
                        st.markdown("#### Detected Emotions")
                        create_emotion_bar_chart(selected_item["emotions"], key=f"history_{selected_id}_emotions")
                
                # Aspect-based analysis
                if selected_item["aspects"]:
                    with st.expander("Aspect-Based Analysis", expanded=True):
                        st.markdown("#### Aspect-Based Sentiment")
                        create_aspect_sentiment_chart(selected_item["aspects"], key=f"history_{selected_id}_aspects")
                        
                        # Display aspects in a table
                        aspects_df = pd.DataFrame(selected_item["aspects"])
                        st.dataframe(aspects_df)
        
        # Visualization of history trend
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
            
            safe_plotly_chart(fig, key="history_trend")
            
        # Add option to download full history as CSV
        if len(history_df) > 0:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download History as CSV",
                data=csv,
                file_name="sentiment_analysis_history.csv",
                mime="text/csv",
            )
            
        # Clear history button
        if st.button("Clear All History"):
            try:
                # Clear history from database
                deleted_count = delete_all_analyses()
                st.session_state.history = []
                st.success(f"Successfully cleared {deleted_count} analysis records from database.")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing history from database: {str(e)}")
                # Still clear from session state
                st.session_state.history = []
                st.rerun()

with tab4:
    # Analytics tab with animated card
    st.markdown("""
    <div class="card">
        <h3><span class="emoji-icon">📈</span> Sentiment Analytics Dashboard</h3>
        <p>Visualize sentiment patterns, emotion trends, and aspect analyses across all your analyzed texts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.history:
        # Empty state with friendly message
        st.markdown("""
        <div style="text-align: center; padding: 40px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;">
            <div style="font-size: 48px; margin-bottom: 20px;">📊</div>
            <h3>No Analytics Data Available</h3>
            <p>Analytics are generated from your analysis history.</p>
            <p>Start by analyzing some text in the Text Analysis tab to populate this dashboard!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create a dataframe from history for analysis
        analytics_data = []
        emotions_data = []
        aspects_data = []
        
        for item in st.session_state.history:
            # Handle different data formats
            if "sentiment" in item:
                # Direct analysis format
                analytics_data.append({
                    "Timestamp": item["timestamp"],
                    "Sentiment": item["sentiment"]["classification"],
                    "Score": item["sentiment"]["compound"],
                    "Positive": item["sentiment"]["positive"],
                    "Negative": item["sentiment"]["negative"],
                    "Neutral": item["sentiment"]["neutral"],
                    "Subjectivity": item["sentiment"]["subjectivity"],
                    "Confidence": item["sentiment"]["confidence"]
                })
            else:
                # Database format (limited analytics data available)
                analytics_data.append({
                    "Timestamp": item["timestamp"],
                    "Sentiment": item["sentiment_classification"],
                    "Score": item["sentiment_score"],
                    "Positive": 0.0,  # Default value
                    "Negative": 0.0,  # Default value
                    "Neutral": 0.0,   # Default value
                    "Subjectivity": 0.5,  # Default value
                    "Confidence": item["confidence"]
                })
            
            # Collect emotions data (safely handle both formats)
            if item["emotions"]:
                for emotion, score in item["emotions"].items():
                    emotions_data.append({
                        "Timestamp": item["timestamp"],
                        "Emotion": emotion,
                        "Score": score
                    })
            
            # Collect aspect data (safely handle both formats)
            if item["aspects"]:
                for aspect in item["aspects"]:
                    # Handle the case where aspect might be a string or dict
                    if isinstance(aspect, dict):
                        aspects_data.append({
                            "Timestamp": item["timestamp"],
                            "Aspect": aspect["aspect"],
                            "Sentiment": aspect["sentiment"],
                            "Score": aspect["score"]
                        })
                    elif isinstance(aspect, str):
                        # Simple default for string-only aspect
                        aspects_data.append({
                            "Timestamp": item["timestamp"],
                            "Aspect": aspect,
                            "Sentiment": "Neutral",
                            "Score": 0.0
                        })
        
        analytics_df = pd.DataFrame(analytics_data)
        emotions_df = pd.DataFrame(emotions_data)
        aspects_df = pd.DataFrame(aspects_data)
        
        # Overall sentiment distribution
        st.subheader("Sentiment Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment classification pie chart
            sentiment_counts = analytics_df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            
            fig = px.pie(
                sentiment_counts,
                values="Count",
                names="Sentiment",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#38B2AC",
                    "Neutral": "#718096",
                    "Negative": "#FF6B6B"
                },
                title="Sentiment Distribution",
                hole=0.4,  # Creates a donut chart effect
            )
            
            # Enhance the chart with 3D effects and interactivity
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>" +
                              "Count: %{value}<br>" +
                              "Percentage: %{percent:.1%}<extra></extra>",
                pull=[0.1, 0.1, 0.1],  # Pulls slices slightly apart
                marker=dict(
                    line=dict(color='#000000', width=2)
                ),
            )
            
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                title_x=0.5,
                title_font_size=20,
                uniformtext_minsize=12,
                uniformtext_mode='hide',
                annotations=[
                    dict(
                        text='Sentiment<br>Analysis',
                        x=0.5,
                        y=0.5,
                        font_size=14,
                        showarrow=False
                    )
                ]
            )
            
            safe_plotly_chart(fig, key="sentiment_distribution_pie")
        
        with col2:
            # Sentiment score histogram
            fig = px.histogram(
                analytics_df,
                x="Score",
                nbins=20,
                color_discrete_sequence=["#5B61F9"],
                title="Sentiment Score Distribution"
            )
            
            fig.update_layout(
                xaxis_title="Sentiment Score (-1 to 1)",
                yaxis_title="Frequency",
                xaxis_range=[-1, 1]
            )
            
            safe_plotly_chart(fig, key="sentiment_histogram")
        
        # Emotion analysis
        if len(emotions_df) > 0:
            st.subheader("Emotion Analysis")
            
            # Aggregate emotions
            emotion_avg = emotions_df.groupby("Emotion")["Score"].mean().reset_index()
            emotion_avg = emotion_avg.sort_values("Score", ascending=False)
            
            fig = px.bar(
                emotion_avg,
                x="Emotion",
                y="Score",
                color="Emotion",
                title="Average Emotion Intensity"
            )
            
            fig.update_layout(
                xaxis_title="Emotion",
                yaxis_title="Average Intensity",
                yaxis_range=[0, 1]
            )
            
            safe_plotly_chart(fig, key="emotion_intensity_bar")
        
        # Aspect analysis
        if len(aspects_df) > 0:
            st.subheader("Aspect Analysis")
            
            # Get the top aspects by frequency
            top_aspects = aspects_df["Aspect"].value_counts().nlargest(10).reset_index()
            top_aspects.columns = ["Aspect", "Frequency"]
            
            # Calculate average sentiment for each aspect
            aspect_sentiment = aspects_df.groupby("Aspect")["Score"].mean().reset_index()
            aspect_sentiment = aspect_sentiment.rename(columns={"Score": "Avg_Sentiment"})
            
            # Merge to get top aspects with their average sentiment
            top_aspects_with_sentiment = top_aspects.merge(aspect_sentiment, on="Aspect")
            
            # Sort by frequency
            top_aspects_with_sentiment = top_aspects_with_sentiment.sort_values("Frequency", ascending=False)
            
            # Create a double-axis chart
            fig = go.Figure()
            
            # Add bars for frequency
            fig.add_trace(go.Bar(
                x=top_aspects_with_sentiment["Aspect"],
                y=top_aspects_with_sentiment["Frequency"],
                name="Frequency",
                marker_color="#5B61F9"
            ))
            
            # Add line for sentiment
            fig.add_trace(go.Scatter(
                x=top_aspects_with_sentiment["Aspect"],
                y=top_aspects_with_sentiment["Avg_Sentiment"],
                name="Average Sentiment",
                marker_color="#FF6B6B",
                yaxis="y2"
            ))
            
            # Set up layout with dual y-axes
            fig.update_layout(
                title="Top Aspects by Frequency and Their Average Sentiment",
                yaxis=dict(title="Frequency"),
                yaxis2=dict(
                    title="Avg Sentiment",
                    overlaying="y",
                    side="right",
                    range=[-1, 1]
                ),
                xaxis_title="Aspect",
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            safe_plotly_chart(fig, key="aspect_frequency_chart")
            
            # Show aspects by sentiment
            st.subheader("Aspects by Sentiment")
            
            # Group aspects by sentiment
            aspect_sentiment_class = aspects_df.groupby(["Aspect", "Sentiment"]).size().reset_index(name="Count")
            
            # Get top aspects
            top_aspects_list = top_aspects["Aspect"].tolist()[:8]  # Limit to top 8 for clarity
            filtered_aspects = aspect_sentiment_class[aspect_sentiment_class["Aspect"].isin(top_aspects_list)]
            
            # Create a grouped bar chart
            fig = px.bar(
                filtered_aspects,
                x="Aspect",
                y="Count",
                color="Sentiment",
                title="Sentiment Analysis by Aspect",
                color_discrete_map={
                    "Positive": "#38B2AC",
                    "Neutral": "#718096",
                    "Negative": "#FF6B6B"
                }
            )
            
            fig.update_layout(
                xaxis_title="Aspect",
                yaxis_title="Count"
            )
            
            safe_plotly_chart(fig, key="aspect_sentiment_chart")
        
        # Subjectivity Analysis
        st.subheader("Subjectivity Analysis")
        
        fig = px.scatter(
            analytics_df,
            x="Score",
            y="Subjectivity",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#38B2AC",
                "Neutral": "#718096",
                "Negative": "#FF6B6B"
            },
            title="Sentiment vs. Subjectivity",
            hover_data=["Timestamp"]
        )
        
        fig.update_layout(
            xaxis_title="Sentiment Score (-1 to 1)",
            yaxis_title="Subjectivity (0-1)",
            xaxis_range=[-1, 1],
            yaxis_range=[0, 1]
        )
        
        safe_plotly_chart(fig, key="subjectivity_scatter")

# Close the tabs container
st.markdown('</div>', unsafe_allow_html=True)

# Footer with information and copyright
st.markdown("""
<div class="footer">
    <h3>About This Dashboard</h3>
    <p>This dashboard uses advanced NLP techniques to analyze sentiment and emotions in text.</p>
    <div class="card" style="margin-bottom: 20px">
        <ul>
            <li><strong>Sentiment Score</strong>: Ranges from -1 (very negative) to 1 (very positive)</li>
            <li><strong>Emotions</strong>: Detects joy, sadness, anger, fear, surprise, and more</li>
            <li><strong>Aspect Analysis</strong>: Extracts topics and their associated sentiments</li>
            <li><strong>Subjectivity</strong>: Measures how subjective (opinion-based) the text is</li>
        </ul>
    </div>
    <p class="copyright">© 2025 Sentiment Analysis Dashboard. All rights reserved.</p>
    <p>Built with ❤️ using Python, NLTK, TextBlob, Streamlit and Plotly</p>
</div>
""", unsafe_allow_html=True)