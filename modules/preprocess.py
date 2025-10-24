import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(data_path):
    try:
        df = pd.read_csv(data_path, index_col=False)
    except Exception as e:
        raise FileNotFoundError(f'File not found: {e}')
        return None
    
    return df

def preprocess_df(df):
    df = df.drop(['Unnamed: 32', 'id'], axis=1)

    numeric_columns = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
        'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
        'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
        'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
        'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst'
    ]

    target_column = 'diagnosis'

    X = df[numeric_columns]
    y = df[target_column]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)   # M/B to 1/0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



    