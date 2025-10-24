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
