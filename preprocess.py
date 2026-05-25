import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle

def preprocess_data(train_path, test_path):
    # Only drop non-numeric/metadata columns that definitely aren't useful for classification
    # or columns that are target-related but not the main label.
    columns_to_drop = ["attack_cat", "id"] # 'id' is just a row index

    train_df = pd.read_csv(train_path).drop(columns=columns_to_drop, errors='ignore')
    test_df = pd.read_csv(test_path).drop(columns=columns_to_drop, errors='ignore')

    # Separate labels
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    x_train = train_df.drop('label', axis=1)
    x_test = test_df.drop('label', axis=1)

    # Identify categorical and numerical columns dynamically
    categorical_cols = x_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = x_train.select_dtypes(exclude=['object']).columns.tolist()

    # Use StandardScaler instead of MinMaxScaler to handle outliers better
    # Use handle_unknown='ignore' for OneHotEncoder to be safe with test data
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )

    x_train_preprocessed = preprocessor.fit_transform(x_train)
    x_test_preprocessed = preprocessor.transform(x_test)

    # Shuffle to ensure classes are mixed
    x_train_preprocessed, y_train = shuffle(x_train_preprocessed, y_train, random_state=42)
    x_test_preprocessed, y_test = shuffle(x_test_preprocessed, y_test, random_state=42)

    return x_train_preprocessed, y_train, x_test_preprocessed, y_test
