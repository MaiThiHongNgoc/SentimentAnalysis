import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # Accuracy evaluation
from sklearn import datasets  # Import datasets
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# 1. Load the 20 Newsgroups dataset
# Using 'sci.med' and 'rec.sport.baseball' categories for sentiment analysis
data = datasets.fetch_20newsgroups(subset='all', categories=['sci.med', 'rec.sport.baseball'])

# 2. Preprocess the data
df = pd.DataFrame({'text': data.data, 'label': data.target})  # Create DataFrame from dataset
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords and non-alpha tokens
    return ' '.join(filtered_tokens)

df['processed_text'] = df['text'].apply(preprocess_text)  # Apply preprocessing

# 3. Split the data into training and testing sets (75% training, 25% testing)
X = df['processed_text']  # Text data
y = df['label']  # Labels (0 or 1 for binary classification, or more for multi-class)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Vectorize the text data (Convert text to numerical data)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_counts = vectorizer.transform(X_test)  # Transform test data

# 5. Build and train the Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# 6. Evaluate the model on the test data
y_pred = clf.predict(X_test_counts)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")  # Print accuracy

# 7. Save the trained model and the vectorizer to pickle files
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
