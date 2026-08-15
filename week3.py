

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string



df = pd.read_csv("spam.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())



print("\nMissing Values:")
print(df.isnull().sum())

df.dropna(inplace=True)



print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning:", df.shape)



print("\nClass Distribution:")
print(df["label"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(x="label", data=df)

plt.title("Spam vs Ham Distribution")
plt.xlabel("Message Type")
plt.ylabel("Count")

plt.show()



def preprocess(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text

df["clean_message"] = df["message"].apply(preprocess)

print("\nOriginal Message:")
print(df["message"].iloc[0])

print("\nCleaned Message:")
print(df["clean_message"].iloc[0])


X = df["clean_message"]

y = df["label"].map({
    "ham":0,
    "spam":1
})

print("\nEncoded Labels:")
print(y.head())


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:",len(X_train))
print("Testing Samples:",len(X_test))


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)

print("\nTF-IDF Completed")
print(X_train.shape)



from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("\nActual:")
print(y_test.values)

print("\nPredicted:")
print(y_pred)



from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

recall = recall_score(y_test,y_pred)

f1 = f1_score(y_test,y_pred)

print("\nAccuracy :",accuracy)
print("Precision:",precision)
print("Recall   :",recall)
print("F1 Score :",f1)

print("\nClassification Report")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham","Spam"]
))



cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham","Spam"],
    yticklabels=["Ham","Spam"]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()



def predict_spam(message):

    message = preprocess(message)

    vector = tfidf.transform([message])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    if prediction==1:
        print("\nMessage:",message)
        print("Prediction : SPAM")
    else:
        print("\nMessage:",message)
        print("Prediction : NOT SPAM")

    print("Probability:",probability)



predict_spam(
"Congratulations! You won a free iPhone. Click here."
)

predict_spam(
"Hey, shall we meet after class?"
)

predict_spam(
"URGENT! Claim your cash reward now."
)

predict_spam(
"Can you send me today's notes?"
)



print("\n=================================")
print("FINAL RESULT")
print("=================================")

print("Accuracy :",accuracy)
print("Precision:",precision)
print("Recall   :",recall)
print("F1 Score :",f1)