# K-Nearest Neighbors (KNN) Student Performance Prediction

This project demonstrates a simple **K-Nearest Neighbors (KNN)** implementation to predict whether a student will **Pass** or **Fail** based on their **Study Hours** and **Sleep Hours**.

---

## 📊 Dataset

The dataset is manually created for simplicity:

| Study Hours | Sleep Hours | Result |
|-------------|-------------|--------|
| 8           | 4           | Fail   |
| 2           | 5           | Fail   |
| 3           | 6           | Fail   |
| 4           | 7           | Pass   |
| 5           | 8           | Pass   |
| 6           | 9           | Pass   |
| 7           | 6           | Pass   |
| 8           | 7           | Pass   |
| 9           | 8           | Pass   |
| 10          | 9           | Pass   |
| 4           | 6           | Fail   |

---

## ⚙️ Steps in the Code

### 1. Data Visualization
The dataset is visualized with **Study Hours** on the X-axis and **Sleep Hours** on the Y-axis. Points are color-coded:
- 🟢 Green = Pass
- 🔴 Red = Fail

### 2. Data Splitting
The dataset is split into **training** (60%) and **testing** (40%) sets using `train_test_split`.

### 3. KNN Model Training
- Uses `KNeighborsClassifier` from scikit-learn.
- `k=3` (3 nearest neighbors).
- The model is trained with the training set.

### 4. Evaluation
- Predictions are made on the test set.
- Accuracy is calculated using `accuracy_score`.

### 5. Prediction for a New Student
- You can input custom `StudyHours` and `SleepHours`.
- The model predicts whether the student will **Pass** or **Fail**.

---

## 🚀 Example Output

```
Training set size: 6 (55%)
Test set size: 5 (45%)

Step 3 (cont.): Model Evaluation
Predictions on test set: ['Pass' 'Fail' 'Pass' 'Pass' 'Pass']
Actual values: ['Pass', 'Fail', 'Pass', 'Pass', 'Fail']
Accuracy: 0.80

Step 4: Prediction for a New Student
New Student [3, 5] is predicted to: Fail
```

---

## 🛠️ Libraries Used
- `pandas`
- `matplotlib`
- `scikit-learn`

---

## 📌 How to Run
1. Install the required libraries:
   ```bash
   pip install pandas matplotlib scikit-learn
   ```
2. Run the Python script.
3. Enter values for **StudyHours** and **SleepHours** when prompted.

---

✅ This project helps beginners understand **KNN classification** with a simple, practical example.
