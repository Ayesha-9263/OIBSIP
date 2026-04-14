# ===============================
# STEP 1: Importing required librariries for data analysis and machine learning
# ===============================

import pandas as pd              # For data handling
import numpy as np               # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns            # For visualization

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


# ===============================
# STEP 2: Load Dataset
# ===============================

# Load your Iris.csv file
iris_data = pd.read_csv("Iris.csv")
print("Unique Species:",iris_data["Species"].unique())

# Show first 5 rows
print(iris_data.head())


# ===============================
# STEP 3: Understand Data
# ===============================

print("\nDataset Info:")
print(iris_data.info())

print("\nStatistical Summary:")
print(iris_data.describe())

print("\nCheck Null Values:")
print(iris_data.isnull().sum())


# ===============================
# STEP 4: Data Preprocessing
# ===============================

# Drop unnecessary column (Id)
iris_data = iris_data.drop("Id", axis=1)

# Convert categorical data (Species) into numbers
le = LabelEncoder()
iris_data['Species'] = le.fit_transform(iris_data['Species'])

print("\nAfter Encoding:")
print(iris_data.head())


# ===============================
# STEP 5: Visualization (Optional but good for assignment)
# ===============================
#Visualizing the relationships between features and species using pairplot

sns.pairplot(iris_data, hue="Species", diag_kind="hist")
plt.suptitle("Feature Relationships in Iris Dataset", y=1.02)
plt.show()


# ===============================
# STEP 6: Split Data
# ===============================

# Features (X) and Target (y)
Features = iris_data.drop("Species", axis=1)
Target = iris_data["Species"]

# Split into training and testing (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    Features, Target, test_size=0.2, random_state=42
)

print("\nTraining Data Size:", x_train.shape)
print("Testing Data Size:", x_test.shape)


# ===============================
# STEP 7: Fitting the model on training data
# ===============================

# Using Decision Tree (simple and effective)
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(x_train, y_train)


# ===============================
# STEP 8: Prediction
# ===============================

y_pred = model.predict(x_test)


# ===============================
# STEP 9: Evaluation
# ===============================

acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy is:{acc:.2f}")
print("\nAccuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# STEP 10: Test with New Data
# ===============================

# Example flower input:
# [sepal_length, sepal_width, petal_length, petal_width]

sample_flower = [[6.0, 3.0, 4.5, 1.5]] #This is a sample input for testing the model, you can change the values to test with diffrent flowers

prediction = model.predict(sample_flower)

# Convert back to original name
print("\nPredicted Flower:", le.inverse_transform(prediction))