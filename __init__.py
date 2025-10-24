from modules.preprocess import load_data, preprocess_df
from modules.train import train
from modules.evaluate import load_model, evaluate_model
import warnings
warnings.filterwarnings('ignore')

def main(data_path):
    df = load_data(data_path)
    if df is None:
        print("Data loading failed.")
        return
    X_train, X_test, y_train, y_test = preprocess_df(df=df)

    model_path = train(X_train=X_train, y_train=y_train)
    
    model = load_model(model_path=model_path)

    evaluate_model(X_test=X_test, y_test=y_test, model=model)


if __name__=="__main__":
    main('data/data.csv')