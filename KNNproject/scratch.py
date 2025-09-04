# All necessary libraries are imported at the beginning
import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt


# Create the dataset
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 4],
    "SleepHours": [4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 6],
    "Result": ["Fail", "Fail", "Fail", "Pass", "Pass",
               "Pass", "Pass", "Pass", "Pass", "Pass", "Fail"]
}
df = pd.DataFrame(data)
print("Step 1: Dataset")
#print(df)

# Visualize the data for initial understanding
print(" Step 2: Visualizing Dataset")
for result, color in [("Pass", "green"), ("Fail", "red")]:
    subset = df[df["Result"] == result]
    plt.scatter(subset["StudyHours"], subset["SleepHours"],
                color=color, label=result, s=100, edgecolor="k")

plt.xlabel("Study Hours")
plt.ylabel("Sleep Hours")
plt.title("Student Performance by Study & Sleep Hours")
plt.legend()
plt.grid(True)
plt.show()

## Step 3: Define KNN Functions

def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def knn_predict(X_train, y_train, x_test, k=3):
    """Predicts the label for a single test point using KNN."""
    distances = []
    for xi, label in zip(X_train, y_train):
        dist = euclidean_distance(xi, x_test)
        distances.append((dist, label))

    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:k]]
    most_common = Counter(k_nearest).most_common(1)[0][0]
    return most_common

## Step 4: Model Training and Evaluation

# Prepare the data
X = df[["StudyHours", "SleepHours"]].values.tolist()
y = df["Result"].tolist()

# Split the data into training and testing sets
X_train, y_train = X[:7], y[:7]
X_test, y_test = X[7:], y[7:]
print(" Step 4: Training & Test Split")
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))

# Make predictions on the test set
predictions = []
for x in X_test:
    pred = knn_predict(X_train, y_train, x, k=3)
    predictions.append(pred)

def accuracy(y_true, y_pred):
    """Calculates the accuracy of the predictions."""
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true)

print(" Step 4 (cont.): Model Evaluation")
print("Predictions on test set:", predictions)
print("Actual values:", y_test)
print("Accuracy:", accuracy(y_test, predictions))

#Step 5: Prediction for a New Student


study_hours_input = float(input("Enter the new student's study hours: "))
sleep_hours_input = float(input("Enter the new student's sleep hours: "))

new_student = [study_hours_input, sleep_hours_input]
prediction = knn_predict(X_train, y_train, new_student, k=3)

print(f"New Student {new_student} is predicted to: {prediction}")


