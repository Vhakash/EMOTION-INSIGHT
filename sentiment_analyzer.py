import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from textblob import TextBlob
import random
import re

# Download necessary NLTK resources
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('punkt')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# Initialize NLTK SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# No transformer models for now due to dependency issues
emotion_classifier = None

def perform_basic_sentiment_analysis(text):
    """
    Perform basic sentiment analysis using NLTK's VADER and TextBlob
    Returns sentiment scores and classification
    """
    # VADER sentiment analysis
    vader_scores = sia.polarity_scores(text)
    
    # TextBlob for additional analysis
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    textblob_subjectivity = blob.sentiment.subjectivity
    
    # Average the compound scores from both approaches
    compound_score = (vader_scores['compound'] + textblob_polarity) / 2
    
    # Determine sentiment classification
    if compound_score >= 0.05:
        classification = "Positive"
    elif compound_score <= -0.05:
        classification = "Negative"
    else:
        classification = "Neutral"
    
    # Calculate confidence based on the strength of the sentiment
    confidence = abs(compound_score) + 0.5  # Scale to 0.5-1.5 range
    confidence = min(1.0, confidence)       # Cap at 1.0
    
    return {
        "compound": compound_score,
        "positive": vader_scores["pos"],
        "negative": vader_scores["neg"],
        "neutral": vader_scores["neu"],
        "subjectivity": textblob_subjectivity,
        "classification": classification,
        "confidence": confidence
    }

def perform_emotion_analysis(text):
    """
    Detect emotions in text using pre-trained models
    Returns a dictionary of emotions and their scores
    """
    # If Hugging Face model is available, use it
    if emotion_classifier:
        try:
            emotions = emotion_classifier(text)
            
            # Convert to a more usable format
            emotion_dict = {}
            for emotion_data in emotions[0]:
                emotion = emotion_data['label']
                score = emotion_data['score']
                emotion_dict[emotion] = score
                
            return emotion_dict
            
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            # Fall back to the basic approach
            pass
    
    # Basic emotion detection fallback using lexicon-based approach
    emotions = {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "neutral": 0.0
    }
    
    # Simple lexicon-based approach
    joy_words = ["happy", "joy", "delighted", "excited", "glad", "pleased", "thrilled"]
    sadness_words = ["sad", "unhappy", "disappointed", "depressed", "upset", "miserable"]
    anger_words = ["angry", "furious", "enraged", "annoyed", "irritated", "frustrated"]
    fear_words = ["afraid", "scared", "frightened", "terrified", "worried", "anxious"]
    surprise_words = ["surprised", "amazed", "astonished", "shocked", "stunned"]
    
    text_lower = text.lower()
    
    # Count occurrences of emotion words
    for word in joy_words:
        if word in text_lower:
            emotions["joy"] += 0.2
    
    for word in sadness_words:
        if word in text_lower:
            emotions["sadness"] += 0.2
    
    for word in anger_words:
        if word in text_lower:
            emotions["anger"] += 0.2
    
    for word in fear_words:
        if word in text_lower:
            emotions["fear"] += 0.2
    
    for word in surprise_words:
        if word in text_lower:
            emotions["surprise"] += 0.2
    
    # Cap values at 1.0 and ensure sum is 1.0
    for emotion in emotions:
        emotions[emotion] = min(1.0, emotions[emotion])
    
    total = sum(emotions.values())
    if total == 0:
        emotions["neutral"] = 1.0
    else:
        emotions["neutral"] = max(0, 1.0 - total)
    
    return emotions

def extract_aspects(text):
    """
    Extract potential aspects/topics from text
    Uses simple NLP techniques to identify noun phrases
    """
    # Use TextBlob for noun phrase extraction
    blob = TextBlob(text)
    noun_phrases = blob.noun_phrases
    
    # Use NLTK for sentence tokenization and POS tagging
    sentences = nltk.sent_tokenize(text)
    
    # Extract aspects and their contexts
    aspects = []
    
    # If no noun phrases found, try to extract nouns from sentences
    if not noun_phrases:
        try:
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            
            # Extract nouns (NN, NNS, NNP, NNPS)
            nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
            
            for noun in nouns:
                if len(noun) > 2:  # Filter out short nouns
                    aspects.append(noun)
    else:
        aspects = [np for np in noun_phrases]
    
    # Remove duplicates while preserving order
    unique_aspects = []
    for aspect in aspects:
        if aspect not in unique_aspects and len(aspect) > 2:
            unique_aspects.append(aspect)
    
    # Limit to top 5 aspects
    return unique_aspects[:5]

def perform_aspect_based_analysis(text):
    """
    Perform aspect-based sentiment analysis
    Extracts aspects and determines sentiment for each
    """
    aspects = extract_aspects(text)
    aspects_with_sentiment = []
    
    if not aspects:
        return []
    
    # Split text into sentences for more accurate aspect sentiment
    sentences = nltk.sent_tokenize(text)
    
    for aspect in aspects:
        # Find sentences containing this aspect
        relevant_sentences = [s for s in sentences if aspect.lower() in s.lower()]
        
        if relevant_sentences:
            # Analyze each relevant sentence
            sentiments = [perform_basic_sentiment_analysis(sentence) for sentence in relevant_sentences]
            
            # Calculate average sentiment
            avg_compound = sum(s["compound"] for s in sentiments) / len(sentiments)
            
            # Determine the sentiment classification
            if avg_compound >= 0.05:
                sentiment = "Positive"
            elif avg_compound <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            aspects_with_sentiment.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "score": avg_compound,
                "context": relevant_sentences[0] if relevant_sentences else ""
            })
        else:
            # If no direct mention, use overall sentiment
            overall = perform_basic_sentiment_analysis(text)
            aspects_with_sentiment.append({
                "aspect": aspect,
                "sentiment": overall["classification"],
                "score": overall["compound"],
                "context": "Inferred from overall text"
            })
    
    return aspects_with_sentiment
