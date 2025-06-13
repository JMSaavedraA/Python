# Import necessary libraries
import pandas as pd
import nltk
import string
import re
import os
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pandas as pd
import nltk
import string
import re
import os
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

# Define preprocessing function
def preprocess(text, stemmer, stop_words):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[\d]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = text.split()
    # Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) >= 2]
    return ' '.join(tokens)

def process_text_files(directory, k, language):
    """
    Processes a directory of text files, applying the preprocessing steps,
    retaining only the k most common words across all texts, and returning
    a CountVectorizer object.

    Args:
        directory: The path to the directory containing the text files.
        k: Number of most common words to retain.

    Returns:
        A tuple (X, vectorizer) where X is the matrix representation of the 
        processed texts, and vectorizer is the CountVectorizer used.
    """
    all_texts = []
    filenames = []

    stemmer = SnowballStemmer(language)  # Or "spanish" if your texts are in Spanish
    stop_words = set(stopwords.words(language))
    
    # Collect all texts and filenames
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Process only .txt files
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                processed_text = preprocess(text, stemmer, stop_words)
                all_texts.append(processed_text)    # Append original processed text for later use
                filenames.append(filename[:-4])     # Save the filename without the '.txt' extension

    # Flatten list of lists to a single list and count word frequencies
    all_word_list = [word for text in all_texts for word in text.split()]
    most_common_words = [word for word, _ in Counter(all_word_list).most_common(k)]
    vectorizer = CountVectorizer(min_df=1, vocabulary=most_common_words)
    X = vectorizer.fit_transform(all_texts)
    
    return X, vectorizer, filenames

X, vectorizer, books = process_text_files('./books',50,"english")

# Load CSV file into DataFrame
titulos_df = pd.read_csv('./titulos.csv')

# Create new DataFrame with 'books' as the ID column
df_books = pd.DataFrame({
    'title': titulos_df['title'],  # Assuming the title is under column 'title'
    'author': titulos_df['author']  # Assuming author names are under column 'author'
})

# Convert sparse matrix to dense DataFrame
X_dense = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Merge the dense matrix with df_books using book IDs
df_books_full = df_books.join(X_dense)

