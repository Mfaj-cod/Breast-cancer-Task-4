import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

def load_model(model_path):
    try:
        with open(model_path, 'rb') as model:
            model = pickle.load(model)
    except Exception as e:
        raise FileNotFoundError(f'Model not found: {e}')
    
    return model

def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_pred) 
    cm = confusion_matrix(y_test, y_pred)

    # Ensuring 'reports' directory exists
    reports_dir = os.path.join(os.getcwd(), 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    pdf_path = os.path.join(reports_dir, 'report.pdf')

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size

        txt = (
            f"Model: LogisticRegression\n\n"
            f"Precision: {precision:.4f}\n\n"
            f"Recall: {recall:.4f}\n\n"
            f"f1_score: {f1:.4f}\n\n"
            f"ROC-AUC score: {rocauc:.4f}\n\n"
            f"Confusion_matrix: \n\n{cm}\n"
        )

        plt.text(
            0.05, 0.95, txt,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            family='monospace'
        )
        plt.title("Model Performance Report")

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Report saved successfully at: {pdf_path}")
