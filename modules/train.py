from sklearn.linear_model import LogisticRegression
import os
import pickle

def train(X_train, y_train):
    model = LogisticRegression(solver='saga', penalty='l2')
    model.fit(X_train, y_train)

    artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, 'model.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model_path