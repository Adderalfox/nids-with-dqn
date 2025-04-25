import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle

def preprocess_data(train_path, test_path):
    columns_to_drop = [
        "attack_cat", "stcpb", "dtcpb", "swin", "dwin",
        "tcprtt", "synack", "ackdat", "ct_flw_http_mthd",
        "smean", "dmean", "sloss", "dloss"
    ]

    train_df = pd.read_csv(train_path).drop(columns=columns_to_drop, errors='ignore')
    test_df = pd.read_csv(test_path).drop(columns=columns_to_drop, errors='ignore')

    y_train = train_df['label'].values
    y_test = test_df['label'].values
    x_train = train_df.drop('label', axis=1)
    x_test = test_df.drop('label', axis=1)

    categorical_cols = ['state', 'proto', 'service']
    numerical_cols = [col for col in x_train.columns if col not in categorical_cols]

    important_protos = ['tcp', 'udp', 'icmp']
    for df in [x_train, x_test]:
        df['proto'] = df['proto'].apply(lambda x:x if x in important_protos else 'others')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', MinMaxScaler(), numerical_cols)
        ]
    )

    x_train_preprocessed = preprocessor.fit_transform(x_train)
    x_test_preprocessed = preprocessor.transform(x_test)

    x_train_preprocessed, y_train = shuffle(x_train_preprocessed, y_train, random_state=42)
    x_test_preprocessed, y_test = shuffle(x_test_preprocessed, y_test, random_state=42)

    return x_train_preprocessed, y_train, x_test_preprocessed, y_test
