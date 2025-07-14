import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Download stopwords if needed
nltk.download('stopwords')

# Load the dataset
spam = pd.read_csv("spam.csv", encoding='ISO-8859-1')
spam = spam[['v1', 'v2']]
spam.columns = ['label', 'message']

# Text preprocessing
ps = PorterStemmer()
corpus = []

for i in range(0, len(spam)):
    review = re.sub('[^a-zA-Z]', ' ', spam['message'][i])
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    corpus.append(' '.join(review))

# Feature extraction
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()
Y = pd.get_dummies(spam['label'])['spam'].values

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train models
model = MultinomialNB()
model.fit(X_train, Y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, pred))
print("Classification Report:\n", classification_report(Y_test, pred))

# Save the model
pickle.dump(model, open("spam_model.pkl", 'wb'))
pickle.dump(cv, open("vectorizer.pkl", 'wb'))

# Test function
def predict_spam(message):
    cleaned = re.sub('[^a-zA-Z]', ' ', message).lower().split()
    cleaned = [ps.stem(word) for word in cleaned if word not in stopwords.words('english')]
    final_message = ' '.join(cleaned)
    vector = cv.transform([final_message]).toarray()
    result = model.predict(vector)
    return "Spam" if result[0] == 1 else "Not Spam"

# Sample test
print(predict_spam("Congratulations! You've won a free ticket to Bahamas!"))
