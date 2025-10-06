# main.py
"""
Machine Learning Model Implementation
-------------------------------------
This script implements a simple Spam Detection model using scikit-learn.
It uses the SMS Spam Collection dataset for demonstration.
"""

# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# 2️⃣ Preprocessing
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# 3️⃣ Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.2, random_state=42
)

# 4️⃣ Convert Text to Features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5️⃣ Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6️⃣ Predictions
y_pred = model.predict(X_test_vec)

# 7️⃣ Evaluation
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 8️⃣ Try on a New Message
sample = ["Congratulations! You've won a free ticket!"]
sample_vec = vectorizer.transform(sample)
print("\n💬 Prediction for sample:", model.predict(sample_vec)[0])
