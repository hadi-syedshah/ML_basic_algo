from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd    
data = {
    "StudyHours": [8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 4],
    "SleepHours": [4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 6],
    "Result": ["Fail", "Fail", "Fail", "Pass", "Pass",
               "Pass", "Pass", "Pass", "Pass", "Pass", "Fail"]
}
df = pd.DataFrame(data)
# Visualize the data for initial understanding
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


##  Model Training and Evaluation

X = df[["StudyHours", "SleepHours"]]
y = df["Result"]

# 2. Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# 3. Now you can use X_train and X_test
print(f"Training set size: {len(X_train)} ({len(X_train) / len(df):.0%})")
print(f"Test set size: {len(X_test)} ({len(X_test) / len(df):.0%})")
# Create and train the KNN model
# n_neighbors is the 'k' value
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nStep 3 (cont.): Model Evaluation")
print("Predictions on test set:", y_pred)
print("Actual values:", list(y_test))
print(f"Accuracy: {accuracy:.2f}")

## Step 4: Prediction for a New Student (with user input)

print("\nStep 4: Prediction for a New Student")

try:
    # Get user input for the new student's data
    study_hours_input = 3#float(input("Enter the new student's study hours: "))
    sleep_hours_input = 5#float(input("Enter the new student's sleep hours: "))

    # Create a new DataFrame with the same feature names as the training data
    new_student_df = pd.DataFrame([[study_hours_input, sleep_hours_input]], columns=['StudyHours', 'SleepHours'])

    # Make the prediction using the new DataFrame
    prediction = knn.predict(new_student_df)

    print(f"New Student {[study_hours_input, sleep_hours_input]} is predicted to: {prediction[0]}")

except ValueError:
    print("Invalid input. Please enter a valid number.")