# 1. Data Loading and Preprocessing
# Data Loading: The dataset is loaded using the Pandas library.
# Text Preprocessing: The text data is tokenized and converted to lowercase.
  
import pandas as pd
import nltk
import sklearn

# Load the dataset
data = pd.read_csv('chatbot_dataset.csv')

# Preprocess the data
nltk.download('punkt')
data['Question'] = data['Question'].apply(lambda x:' '.join(nltk.word_tokenize(x.lower())))
print(data.head())  # .head is thats why first 5 row will be printed

# 2. Vectorizing Text Data

# We convert the text data into numerical values using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Question'])
print(X.shape)

# Training a Text Classification Model

# We use the Naive Bayes classifier to train the model on the vectorized text data

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data['Concept'], data['Description'], test_size=0.2, random_state=42)

# Create model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(x_train, y_train)
print("Completed training")

# 4. Implement a Function to get a Chatbot Response
# Chatbot Function: Implement a function to get responses from the chatbot based on the trained model.

# Function to get response from the chatbot
def get_response(question):
    question = ' '.join(nltk.word_tokenize(question.lower()))
    answer = model.predict([question])[0]
    return answer

# Testing the chatbot
print(get_response("What is NLP?"))
