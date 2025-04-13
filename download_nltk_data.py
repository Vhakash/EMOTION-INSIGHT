import nltk
import ssl
from textblob.download_corpora import download_all

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('conll2000')

# Download TextBlob data
print("Downloading TextBlob corpora...")
download_all()

print("All data downloaded successfully!")