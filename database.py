import os
import json
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Set up database URL - using SQLite for local development
DATABASE_URL = "sqlite:///sentiment_analysis.db"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
metadata = MetaData()

# Define SentimentAnalysis model
class SentimentAnalysis(Base):
    __tablename__ = 'sentiment_analyses'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    text = Column(Text)
    sentiment_score = Column(Float)
    sentiment_classification = Column(String(20))
    confidence = Column(Float)
    emotions = Column(Text)  # Store as JSON
    aspects = Column(Text)   # Store as JSON
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "text": self.text,
            "sentiment_score": self.sentiment_score,
            "sentiment_classification": self.sentiment_classification,
            "confidence": self.confidence,
            "emotions": json.loads(self.emotions) if self.emotions else {},
            "aspects": json.loads(self.aspects) if self.aspects else []
        }

# Create tables
def init_db():
    Base.metadata.create_all(engine)

# Create a session factory
Session = sessionmaker(bind=engine)

# Functions to interact with the database
def save_analysis(analysis_result):
    """Save analysis result to database"""
    session = Session()
    
    try:
        # Extract data from analysis result
        new_analysis = SentimentAnalysis(
            timestamp=datetime.strptime(analysis_result["timestamp"], "%Y-%m-%d %H:%M:%S"),
            text=analysis_result["text"],
            sentiment_score=analysis_result["sentiment"]["compound"],
            sentiment_classification=analysis_result["sentiment"]["classification"],
            confidence=analysis_result["sentiment"]["confidence"],
            emotions=json.dumps(analysis_result["emotions"]),
            aspects=json.dumps(analysis_result["aspects"])
        )
        
        session.add(new_analysis)
        session.commit()
        
        return new_analysis.id
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {str(e)}")
        return None
    finally:
        session.close()

def get_all_analyses():
    """Retrieve all analyses from the database"""
    session = Session()
    
    try:
        analyses = session.query(SentimentAnalysis).order_by(SentimentAnalysis.timestamp.desc()).all()
        return [analysis.to_dict() for analysis in analyses]
    except Exception as e:
        print(f"Error retrieving from database: {str(e)}")
        return []
    finally:
        session.close()

def get_analysis_by_id(analysis_id):
    """Retrieve a specific analysis by ID"""
    session = Session()
    
    try:
        analysis = session.query(SentimentAnalysis).filter(SentimentAnalysis.id == analysis_id).first()
        if analysis:
            return analysis.to_dict()
        return None
    except Exception as e:
        print(f"Error retrieving from database: {str(e)}")
        return None
    finally:
        session.close()

def delete_analysis(analysis_id):
    """Delete a specific analysis by ID"""
    session = Session()
    
    try:
        analysis = session.query(SentimentAnalysis).filter(SentimentAnalysis.id == analysis_id).first()
        if analysis:
            session.delete(analysis)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"Error deleting from database: {str(e)}")
        return False
    finally:
        session.close()

def get_sentiment_distribution():
    """Get distribution of sentiment classifications"""
    session = Session()
    
    try:
        analyses = session.query(SentimentAnalysis).all()
        distribution = {"Positive": 0, "Negative": 0, "Neutral": 0}
        
        for analysis in analyses:
            if analysis.sentiment_classification in distribution:
                distribution[analysis.sentiment_classification] += 1
        
        return distribution
    except Exception as e:
        print(f"Error getting sentiment distribution: {str(e)}")
        return {"Positive": 0, "Negative": 0, "Neutral": 0}
    finally:
        session.close()

def get_sentiment_history_dataframe():
    """Get history data as pandas DataFrame for visualization"""
    session = Session()
    
    try:
        analyses = session.query(SentimentAnalysis).order_by(SentimentAnalysis.timestamp).all()
        data = []
        
        for analysis in analyses:
            data.append({
                "Timestamp": analysis.timestamp,
                "Score": analysis.sentiment_score,
                "Classification": analysis.sentiment_classification,
                "Text": analysis.text[:50] + "..." if len(analysis.text) > 50 else analysis.text
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error getting history data: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def delete_all_analyses():
    """Delete all analyses from the database"""
    session = Session()
    
    try:
        count = session.query(SentimentAnalysis).count()
        session.query(SentimentAnalysis).delete()
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        print(f"Error deleting all analyses: {str(e)}")
        return 0
    finally:
        session.close()

# Initialize the database when this module is imported
init_db()