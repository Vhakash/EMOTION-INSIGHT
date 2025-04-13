import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_sentiment_gauge(score, key=None):
    """
    Create a gauge chart to display sentiment score
    
    Parameters:
    score : float
        The sentiment score to display
    key : str, optional
        A unique key for the chart element
    """
    # Define colors and thresholds
    colors = ['#FF6B6B', '#718096', '#38B2AC']  # Negative, Neutral, Positive
    
    # Create the gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score"},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#2D3748"},
            'bar': {'color': "#5B61F9"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.05], 'color': colors[0]},  # Negative
                {'range': [-0.05, 0.05], 'color': colors[1]},  # Neutral
                {'range': [0.05, 1], 'color': colors[2]},  # Positive
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    # Customize layout
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#F8F9FA",
        font=dict(color="#2D3748")
    )
    
    # Display the chart with a unique key
    st.plotly_chart(fig, use_container_width=True, key=f"sentiment_gauge_{key or 'main'}")

def create_emotion_bar_chart(emotions, key=None):
    """
    Create a horizontal bar chart for emotion analysis
    
    Parameters:
    emotions : dict
        A dictionary of emotions and their scores
    key : str, optional
        A unique key for the chart element
    """
    # Convert emotions dict to DataFrame
    emotion_df = pd.DataFrame({
        'Emotion': list(emotions.keys()),
        'Score': list(emotions.values())
    })
    
    # Sort by score in descending order
    emotion_df = emotion_df.sort_values('Score', ascending=False)
    
    # Define a color map for emotions
    color_map = {
        'joy': '#38B2AC',  # Teal
        'sadness': '#718096',  # Grey
        'anger': '#FF6B6B',  # Coral
        'fear': '#805AD5',  # Purple
        'surprise': '#F6E05E',  # Yellow
        'neutral': '#CBD5E0',  # Light Grey
        'disgust': '#F56565',  # Red
        'love': '#FC8181',  # Pink
    }
    
    # Create default colors for any emotion not in the map
    colors = [color_map.get(emotion.lower(), '#5B61F9') for emotion in emotion_df['Emotion']]
    
    # Create the bar chart
    fig = px.bar(
        emotion_df,
        y='Emotion',
        x='Score',
        orientation='h',
        labels={'Score': 'Intensity', 'Emotion': ''},
        text='Score',
        color='Emotion',
        color_discrete_sequence=colors
    )
    
    # Customize layout
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="#F8F9FA",
        plot_bgcolor="#F8F9FA",
        font=dict(color="#2D3748"),
        showlegend=False,
        xaxis=dict(range=[0, 1])
    )
    
    # Format text labels
    fig.update_traces(
        texttemplate='%{x:.2f}',
        textposition='outside'
    )
    
    # Display the chart with a unique key
    st.plotly_chart(fig, use_container_width=True, key=f"emotion_chart_{key or 'main'}")

def create_aspect_sentiment_chart(aspects, key=None):
    """
    Create a visualization for aspect-based sentiment analysis
    
    Parameters:
    aspects : list
        A list of aspect dictionaries with 'aspect', 'sentiment', and 'score' keys
    key : str, optional
        A unique key for the chart element
    """
    # Convert aspects to DataFrame
    aspect_df = pd.DataFrame(aspects)
    
    # Define colors for sentiment
    colors = {
        'Positive': '#38B2AC',  # Teal
        'Neutral': '#718096',  # Grey
        'Negative': '#FF6B6B'   # Coral
    }
    
    # Create a horizontal bar chart
    fig = px.bar(
        aspect_df,
        y='aspect',
        x='score',
        orientation='h',
        labels={'score': 'Sentiment Score', 'aspect': 'Aspect'},
        color='sentiment',
        color_discrete_map=colors,
        text='sentiment'
    )
    
    # Add a vertical line at x=0
    fig.add_shape(
        type='line',
        x0=0, y0=-0.5,
        x1=0, y1=len(aspect_df)-0.5,
        line=dict(color='#2D3748', width=1, dash='dash')
    )
    
    # Customize layout
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="#F8F9FA",
        plot_bgcolor="#F8F9FA",
        font=dict(color="#2D3748"),
        xaxis=dict(
            title='Negative ← → Positive',
            range=[-1, 1],
            zeroline=True,
            zerolinecolor='#2D3748',
            zerolinewidth=1
        )
    )
    
    # Display the chart with a unique key
    st.plotly_chart(fig, use_container_width=True, key=f"aspect_chart_{key or 'main'}")
