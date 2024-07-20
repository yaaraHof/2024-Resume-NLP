import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
import re
import os

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Make sure we're in the currect dir for pd.read_csv
print(os.getcwd())

# Initialize lemmatizer
lemmatizer = nltk.WordNetLemmatizer()

# Load the data
df = pd.read_csv('2024-Resume-NLP/dataset/Resume/resume.csv')

# Print the shape of the original dataset
print("Original dataset shape:", df.shape)

# Sample small part of the dataset
df = df.sample(frac=0.01, random_state=42)

# Drop the original Resume_html column
df = df.drop(columns=['Resume_html'])

# Print the shape of the sampled dataset
print("Sampled dataset shape:", df.shape)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = nltk.word_tokenize(text)  # Tokenize
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    return ' '.join(tokens)

# Apply preprocessing
df['Resume_str'] = df['Resume_str'].apply(preprocess_text)

# Print a clean example
example_index = 0
print("Resume example:\n", df.iloc[example_index])
