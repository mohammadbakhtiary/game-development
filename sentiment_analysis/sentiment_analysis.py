import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "text": [
        "I love this movie, it was fantastic!",
        "This product is terrible, I hate it!",
        "It’s okay, nothing special.",
        "What a great day!",
        "I’m so sad and disappointed.",
        "Absolutely wonderful experience!",
        "Worst experience ever."
    ],
    "label": ["positive", "negative", "neutral", "positive", "negative", "positive", "negative"]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["positive", "neutral", "negative"], yticklabels=["positive", "neutral", "negative"])
plt.title("Confusion Matrix")
plt.show()

new_texts = ["I really like this!", "This was awful.", "It was okay."]
new_vec = vectorizer.transform(new_texts)
predictions = model.predict(new_vec)

for text, label in zip(new_texts, predictions):
    print(f"💬 '{text}' → {label}")
