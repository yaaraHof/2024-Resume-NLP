import pandas as pd
import nltk
import re
import os

# # Download necessary NLTK data (only run once)
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

# # Sample small part of the dataset (only for testing)
# df = df.sample(frac=0.01, random_state=42)

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

# Squeeky-Clean:
# Check for NaN values in the 'Resume_str' column
print("Number of NaN values in 'Resume_str':", df['Resume_str'].isna().sum())
# Drop rows with NaN values in 'Resume_str'
df = df.dropna(subset=['Resume_str'])

# Print a clean example
example_index = 1
print("Aplicant example:\n", df.iloc[example_index])
print("Resume text example:\n", df.iloc[example_index]['Resume_str'])

# Save the cleaned data to a new CSV file
cleaned_csv_path = '2024-Resume-NLP/dataset/Resume/cleaned-Resume.csv'
df.to_csv(cleaned_csv_path, index=False)
print(f"Cleaned data saved to {cleaned_csv_path}")