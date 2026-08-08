import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("Iris.csv")

df = df.drop("Id", axis=1)

df.dropna(inplace=True)
print(df.isnull().sum())

print(df.drop_duplicates())

print(df["Species"].unique())
print(df["Species"].value_counts())

x = df.drop("Species", axis=1)
y = df["Species"]

print(x.head())
print(y.head())



import seaborn as sns


plt.figure(figsize=(7, 5))
sns.countplot(x="Species", data=df)
plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()



sns.pairplot(df, hue="Species", diag_kind="hist")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()



x.hist(figsize=(10, 8), bins=15, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(data=x)
plt.title("Boxplot of Iris Features")
plt.xlabel("Features")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.show()



plt.figure(figsize=(8, 6))
sns.heatmap(x.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)




scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("\nFeature scaling completed.")




from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=200)


logistic_model.fit(x_train_scaled, y_train)


y_pred_logistic = logistic_model.predict(x_test_scaled)

print("\n========================================")
print("LOGISTIC REGRESSION")
print("========================================")

print("Actual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred_logistic)



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_precision = precision_score(
    y_test,
    y_pred_logistic,
    average="weighted"
)
logistic_recall = recall_score(
    y_test,
    y_pred_logistic,
    average="weighted"
)
logistic_f1 = f1_score(
    y_test,
    y_pred_logistic,
    average="weighted"
)

print("\nLogistic Regression Performance:")
print("Accuracy :", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall   :", logistic_recall)
print("F1 Score :", logistic_f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))


cm_logistic = confusion_matrix(y_test, y_pred_logistic)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=logistic_model.classes_,
    yticklabels=logistic_model.classes_
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()



from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


random_forest_model.fit(x_train, y_train)


y_pred_rf = random_forest_model.predict(x_test)

print("\n========================================")
print("RANDOM FOREST")
print("========================================")

print("Actual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred_rf)




rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(
    y_test,
    y_pred_rf,
    average="weighted"
)
rf_recall = recall_score(
    y_test,
    y_pred_rf,
    average="weighted"
)
rf_f1 = f1_score(
    y_test,
    y_pred_rf,
    average="weighted"
)

print("\nRandom Forest Performance:")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))



cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=random_forest_model.classes_,
    yticklabels=random_forest_model.classes_
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()




results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        logistic_accuracy,
        rf_accuracy
    ],
    "Precision": [
        logistic_precision,
        rf_precision
    ],
    "Recall": [
        logistic_recall,
        rf_recall
    ],
    "F1 Score": [
        logistic_f1,
        rf_f1
    ]
})

print("\n========================================")
print("MODEL PERFORMANCE COMPARISON")
print("========================================")

print(results)



results_plot = results.set_index("Model")

results_plot.plot(
    kind="bar",
    figsize=(10, 6),
    edgecolor="black"
)

plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()




feature_importance = pd.DataFrame({
    "Feature": x.columns,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("========================================")

print(feature_importance)


plt.figure(figsize=(9, 6))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()




if rf_accuracy > logistic_accuracy:
    print("\nBest Model: Random Forest")
    print("Best Accuracy:", rf_accuracy)
else:
    print("\nBest Model: Logistic Regression")
    print("Best Accuracy:", logistic_accuracy)