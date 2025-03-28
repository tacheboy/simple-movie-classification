import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

# Ensure stopwords are downloaded
nltk.download('stopwords', quiet=True)

# Global constants for performance optimization
STOPWORDS = set(stopwords.words('english'))
PUNCT_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")
STEMMER = LancasterStemmer()

def clean_text(text: str) -> str:
    """
    Clean text by lowering case, removing punctuation,
    filtering stopwords, and applying stemming.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation using pre-compiled regex
    text = PUNCT_REGEX.sub("", text)
    # Tokenize the text
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]
    # Apply stemming using the pre-created stemmer
    tokens = [STEMMER.stem(word) for word in tokens]
    return ' '.join(tokens)
