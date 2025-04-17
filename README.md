
# Sentiment Analysis Dashboard

A comprehensive web application for analyzing sentiment and emotions in text using advanced NLP techniques. Built with Streamlit, NLTK, TextBlob, and Plotly.

## Features

- **Basic Sentiment Analysis**: Analyzes text sentiment using VADER and TextBlob
- **Emotion Detection**: Identifies emotions like joy, sadness, anger, fear, and surprise
- **Aspect-Based Analysis**: Extracts topics and their associated sentiments
- **Interactive Visualizations**: Includes gauges, charts, and graphs for data representation
- **History Tracking**: Saves all analyses with timestamp for future reference
- **Batch Processing**: Supports CSV file uploads for bulk analysis
- **Analytics Dashboard**: Provides insights across all analyzed texts

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **NLP**: NLTK, TextBlob
- **Database**: SQLAlchemy
- **Visualization**: Plotly
- **Styling**: Custom CSS animations

## Getting Started

1. Click the "Run" button to start the Streamlit server
2. The app will be available at port 5000
3. Use the navigation tabs to access different features:
   - Text Analysis
   - Batch Analysis
   - History
   - Analytics

## Usage

### Single Text Analysis
1. Navigate to "Text Analysis" tab
2. Enter text or select a sample
3. Click "Analyze Sentiment"
4. View results in interactive charts

### Batch Analysis
1. Go to "Batch Analysis" tab
2. Upload a CSV file with a 'text' column
3. Process the file
4. Download results as CSV

### View History
- Access "History" tab to view past analyses
- Select entries to see detailed results
- Export history as CSV

### Analytics
- Use "Analytics" tab for overall insights
- View sentiment distribution
- Analyze emotion trends
- Track aspect-based patterns

## Project Structure

- `app.py`: Main application file
- `sentiment_analyzer.py`: Core sentiment analysis logic
- `visualization.py`: Chart creation functions
- `database.py`: Database operations
- `utils.py`: Helper functions

## Dependencies

All required packages are automatically installed through pyproject.toml:
- nltk
- numpy
- pandas
- plotly
- sqlalchemy
- streamlit
- textblob

## Notes

- The application uses NLTK's VADER lexicon for sentiment analysis
- Emotion detection uses a lexicon-based approach
- Aspect extraction utilizes NLTK's part-of-speech tagging
- All data is stored in a SQLite database
