import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



df = pd.read_csv("Iris.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()


 
df = df.drop("Id", axis=1)

df.dropna(inplace=True)

print("\nNull values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

print("\nDataset after removing duplicates:")
print(df)
 


print("\nUnique Species:")
print(df["Species"].unique())

print("\nSpecies Counts:")
print(df["Species"].value_counts())



plt.figure(figsize=(7, 5))

species_counts = df["Species"].value_counts()

plt.bar(
    species_counts.index,
    species_counts.values
)

plt.title("Number of Flowers in Each Species")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")

plt.xticks(rotation=15)

plt.show()

x = df.drop("Species", axis=1)

y = df["Species"]

print("\nFeatures:")
print(x.head())

print("\nLabels:")
print(y.head())


 
plt.figure(figsize=(8, 6))

for species in df["Species"].unique():

    data = df[df["Species"] == species]

    plt.scatter(
        data["SepalLengthCm"],
        data["SepalWidthCm"],
        label=species
    )

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()

plt.show()
 

 
plt.figure(figsize=(8, 6))

for species in df["Species"].unique():

    data = df[df["Species"] == species]

    plt.scatter(
        data["PetalLengthCm"],
        data["PetalWidthCm"],
        label=species
    )

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.show()
 

 
print("\nCreating Pair Plot...")

sns.pairplot(
    df,
    hue="Species",
    diag_kind="hist"
)

plt.suptitle(
    "Pair Plot - Iris Feature Relationships",
    y=1.02
)

plt.show()


 
print("\nCreating Correlation Heatmap...")

numeric_features = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

correlation = numeric_features.corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Iris Features")

plt.show()



plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Species",
    y="SepalLengthCm"
)

plt.title("Sepal Length Across Iris Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length (cm)")

plt.show()



plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Species",
    y="SepalWidthCm"
)

plt.title("Sepal Width Across Iris Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")

plt.show()




plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Species",
    y="PetalLengthCm"
)

plt.title("Petal Length Across Iris Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")

plt.show()


plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Species",
    y="PetalWidthCm"
)

plt.title("Petal Width Across Iris Species")
plt.xlabel("Species")
plt.ylabel("Petal Width (cm)")

plt.show()


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="PetalLengthCm",
    hue="Species"
)

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()

plt.show()



plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalWidthCm",
    y="PetalWidthCm",
    hue="Species"
)

plt.title("Sepal Width vs Petal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.show()



plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)

plt.title("Sepal Length vs Petal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()

plt.show()


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalWidthCm",
    y="PetalLengthCm",
    hue="Species"
)

plt.title("Sepal Width vs Petal Length")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()

plt.show()

 
print("\n========================================")
print("WEEK 4 - FEATURE ANALYSIS")
print("========================================")

print("""
1. Petal Length and Petal Width show a strong positive
   correlation.

2. The Pair Plot shows that Petal Length and Petal Width
   provide better separation between the three Iris species.

3. The Box Plots show clear differences in petal measurements
   among the three species.

4. Sepal Length and Sepal Width show more overlap between
   the species.

5. Petal Length and Petal Width are the most useful features
   for distinguishing Iris species.
""")



print("\n========================================")
print("WEEK 4 - CONCLUSION")
print("========================================")

print("""
Data visualization was performed using Pair Plots,
Correlation Heatmap, Box Plots, and Scatter Plots.

The visualizations show that Petal Length and Petal Width
have a strong relationship and provide better separation
between the three Iris species.

Sepal Length and Sepal Width have more overlap between
the species.

Therefore, Petal Length and Petal Width are the most
important features for identifying the Iris species.
""")


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:", x_train.shape)
print("Testing data size:", x_test.shape)



model = LogisticRegression(max_iter=200)


model.fit(x_train, y_train)



y_pred = model.predict(x_test)

print("\nPredicted values:")
print(y_pred)



accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")


 
confusion = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(confusion)


 
plt.figure(figsize=(7, 6))

plt.imshow(confusion)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.colorbar()
species_names = model.classes_

plt.xticks(
    range(len(species_names)),
    species_names,
    rotation=15
)

plt.yticks(
    range(len(species_names)),
    species_names
)
 
for i in range(len(confusion)):
    for j in range(len(confusion)):
        plt.text(
            j,
            i,
            confusion[i, j],
            ha="center",
            va="center"
        )

plt.show()


 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


 
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(comparison)


plt.figure(figsize=(10, 5))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted"
)

plt.title("Actual vs Predicted Species")
plt.xlabel("Test Sample")
plt.ylabel("Species")

plt.legend()

plt.xticks(range(len(y_test)))

plt.show()



new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=x.columns
)

prediction = model.predict(new_flower)

print("\nNew Flower Details:")
print(new_flower)

print("\nPredicted Species:", prediction[0])


print("\n========================================")
print("FINAL PROJECT SUMMARY")
print("========================================")

print("""
Dataset: Iris Dataset

Machine Learning Algorithm:
Logistic Regression

Data Visualization:
- Species Count Bar Chart
- Sepal Scatter Plot
- Petal Scatter Plot
- Pair Plot
- Correlation Heatmap
- Box Plots
- Additional Scatter Plots

Important Features:
- Petal Length
- Petal Width

Model Accuracy:
""")

print(accuracy * 100, "%")

print("""
Final Finding:
Petal Length and Petal Width are the most useful features
for distinguishing the three Iris species.
""")