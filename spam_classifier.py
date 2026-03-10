import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Select first two columns
data = data.iloc[:, :2]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test example
email = ["Win a free iPhone now"]

email_tfidf = vectorizer.transform(email)
prediction = model.predict(email_tfidf)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")
