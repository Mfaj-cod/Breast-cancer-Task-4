####🧠 Breast Cancer Detection using Logistic Regression

📘 Project Overview

This project focuses on building a Machine Learning model to classify breast cancer tumors as Malignant (M) or Benign (B) using the Breast Cancer Wisconsin Dataset.

The model uses Logistic Regression, a simple yet powerful classification algorithm, to predict whether a tumor is malignant or benign based on various cell nucleus features such as radius, texture, perimeter, and area.

🚀 Project Pipeline
1️⃣ Data Loading

The dataset (data/data.csv) is loaded using Pandas.

Unnecessary columns such as id and Unnamed: 32 were dropped since they don’t contribute to prediction.

df = pd.read_csv('data/data.csv')
df = df.drop(['Unnamed: 32', 'id'], axis=1)

2️⃣ Data Preprocessing

Preprocessing was handled inside a dedicated function preprocess_df().

Steps:

Feature Selection:
Selected all numeric columns related to cell characteristics such as radius_mean, texture_mean, area_worst, etc.

Target Encoding:
The target column diagnosis had two categorical values:

M → Malignant

B → Benign
These were encoded into binary format using LabelEncoder:

encoder = LabelEncoder()
y = encoder.fit_transform(df['diagnosis'])


Feature Scaling:
Applied StandardScaler to normalize the numeric features.
This ensures that all features contribute equally to the model and prevents scale dominance.

Train-Test Split:
Split data into:

80% training set

20% testing set

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

3️⃣ Model Training

The training process was implemented in the train() function.

Used Logistic Regression as the base model due to its interpretability and robustness for binary classification tasks.

The model was trained on the scaled training data and serialized using pickle for later use.

model = LogisticRegression(solver='saga', penalty='l2')
model.fit(X_train, y_train)


The trained model is saved in the artifacts/ directory as model.pkl.

4️⃣ Model Evaluation

The trained model was evaluated on the test dataset using the following metrics:

Precision

Recall

F1-Score

ROC_AUC_score

Confusion Matrix

A performance report was generated and saved as a PDF in the reports/ directory.

Metric	Score
Precision	1.0000
Recall	0.9767
F1-Score	0.9882
ROC-AUC score   0.9884

Confusion Matrix:

[[71  0]
 [ 1 42]]


✅ Interpretation:

High precision and recall indicate the model performs well at identifying malignant cases while minimizing false predictions.

The confusion matrix shows only 1 misclassifications out of 114 samples — a strong result.

5️⃣ Report Generation

A custom evaluation script was created to automatically:

Compute metrics

Generate report

Export a performance report (report.pdf) using matplotlib.backends.backend_pdf.PdfPages

📂 Directory Structure
Task-4/
│
├── data/
│   └── data.csv
│
├── modules/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── artifacts/
│   └── model.pkl
│
├── reports/
│   └── report.pdf
│
└── __init__.py

🧩 Key Learnings

Difference between linear and logistic regression

Importance of feature scaling and encoding

Evaluating models using precision, recall, F1, and confusion matrix

How to handle the ML pipeline modularly (preprocess → train → evaluate)

Exporting models and reports programmatically

🏁 Conclusion

The logistic regression model achieved high accuracy and reliability in predicting breast cancer malignancy.
Its simplicity and interpretability make it an excellent baseline model for this classification problem.

Future improvements can include experimenting with:

Regularization techniques (L1/L2)

Feature selection methods

Ensemble models (Random Forest, XGBoost)













📘 Logistic Regression — Interview Q&A

1️⃣ How does Logistic Regression differ from Linear Regression?

| Aspect                  | Linear Regression                               | Logistic Regression                                         |
| ----------------------- | ----------------------------------------------- | ----------------------------------------------------------- |
| **Purpose**             | Predicts continuous values (e.g., house prices) | Predicts categorical outcomes (e.g., yes/no, spam/not spam) |
| **Output Range**        | Any real number (−∞ to +∞)                      | Probability between 0 and 1                                 |
| **Activation Function** | None                                            | Uses **Sigmoid (logistic)** function                        |
| **Model Type**          | Regression model                                | Classification model                                        |
| **Error Metric**        | Mean Squared Error (MSE)                        | Log Loss or Cross-Entropy                                   |


💡 Key idea: Logistic regression uses a linear combination of inputs but applies the sigmoid function to map outputs into probabilities.


2️⃣ What is the Sigmoid Function?

The sigmoid function converts any real-valued number into a probability between 0 and 1.
It’s defined as:

𝜎
(
𝑥
)
=
1
1
+
𝑒
−
𝑥
σ(x)=
1+e
−x
1
	​


If x → +∞ → σ(x) ≈ 1

If x → −∞ → σ(x) ≈ 0

In logistic regression, this probability is used to classify samples:

If σ(x) ≥ threshold → class 1
Else → class 0

3️⃣ What is Precision vs Recall?

| Metric                   | Formula        | Meaning                                                      |
| ------------------------ | -------------- | ------------------------------------------------------------ |
| **Precision**            | TP / (TP + FP) | Of all predicted positives, how many were actually positive? |
| **Recall (Sensitivity)** | TP / (TP + FN) | Of all actual positives, how many were correctly predicted?  |

Precision answers: “How accurate are positive predictions?”

Recall answers: “How many actual positives did we catch?”

👉 In medical or fraud detection cases, high recall is often more critical.

4️⃣ What is the ROC-AUC Curve?

ROC (Receiver Operating Characteristic) curve plots:

X-axis: False Positive Rate (FPR)

Y-axis: True Positive Rate (TPR = Recall)

The AUC (Area Under Curve) represents how well the model distinguishes between classes.

AUC = 1.0 → Perfect model

AUC = 0.5 → Random guessing

💡 A higher AUC means the model is better at ranking positives above negatives.

5️⃣ What is the Confusion Matrix?

A confusion matrix summarizes classification performance using four outcomes:

|                     | **Predicted Positive** | **Predicted Negative** |
| ------------------- | ---------------------- | ---------------------- |
| **Actual Positive** | True Positive (TP)     | False Negative (FN)    |
| **Actual Negative** | False Positive (FP)    | True Negative (TN)     |

Example:

[[TN  FP]
 [FN  TP]]


It helps derive metrics like accuracy, precision, recall, and F1-score.

6️⃣ What happens if classes are imbalanced?

When one class heavily outweighs the other (e.g., 95% negative, 5% positive):

The model may get high accuracy by predicting the majority class.

Metrics like Precision, Recall, and AUC become more informative than accuracy.

Solutions include:

Resampling: Oversampling the minority or undersampling the majority class

Class weights: Giving more importance to minority class errors

Synthetic data (SMOTE)

7️⃣ How do you choose the threshold?

By default, logistic regression uses a 0.5 threshold to decide between 0 and 1.
However, you can tune it based on:

Business goal: e.g., prioritize recall (lower threshold) for medical diagnosis

ROC/Precision-Recall curve: Find the point that balances TPR and FPR

F1-score optimization: Choose the threshold maximizing F1

8️⃣ Can Logistic Regression be used for Multi-Class Problems?

✅ Yes!
This is called Multinomial Logistic Regression.

Two main strategies:

One-vs-Rest (OvR): Train a separate classifier for each class vs. all others.

Multinomial (Softmax): Generalizes the logistic function to multiple classes using the softmax function.

Example:

LogisticRegression(multi_class='multinomial', solver='lbfgs')

🧩 Summary Table

| Concept            | Key Takeaway                                          |
| ------------------ | ----------------------------------------------------- |
| Logistic vs Linear | Logistic outputs probabilities, not continuous values |
| Sigmoid            | Maps real numbers → probabilities                     |
| Precision & Recall | Precision = correctness, Recall = completeness        |
| ROC-AUC            | Measures overall separability                         |
| Confusion Matrix   | Summarizes TP, FP, FN, TN                             |
| Imbalanced Classes | Use class weights or resampling                       |
| Threshold          | Adjust based on objective                             |
| Multi-Class        | Use OvR or Softmax extensions                         |
