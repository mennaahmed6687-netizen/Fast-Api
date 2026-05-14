import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):

    df['Who_completed_the_test'] = df['Who_completed_the_test'].replace(
        ["Family Member"], 'Family member'
    )

    replacements = {
        'middle eastern': 'Middle Eastern',
        'mixed': 'Mixed',
        'asian': 'Asian',
        'black': 'Black',
        'south asian': 'South Asian'
    }

    df['Ethnicity'] = df['Ethnicity'].replace(replacements)

    return df

def handle_missing_values(df):


    subset_A = df.dropna(subset=['Qchat_10_Score'])

    subset_B = df[df['Qchat_10_Score'].isnull()].copy()

    imputed_values = np.random.choice(
        subset_A['Qchat_10_Score'],
        size=len(subset_B),
        replace=True
    )

    subset_B.loc[:, 'Qchat_10_Score'] = imputed_values

    df = pd.concat([subset_A, subset_B], ignore_index=True)

    cols = [
        'Depression',
        'Social_Responsiveness_Scale',
        'Social/Behavioural Issues'
    ]

    df[cols] = df[cols].replace('?', np.nan)

    imputer = SimpleImputer(strategy='most_frequent')

    df[cols] = imputer.fit_transform(df[cols])

    return df

def create_result_feature(df):

    df['Result'] = df.iloc[:, 1:11].sum(axis=1)

    return df


def encode_yes_no(df):

    bool_columns = [
        'Social_Responsiveness_Scale',
        'Speech Delay/Language Disorder',
        'Learning disorder',
        'Genetic_Disorders',
        'Depression',
        'Global developmental delay/intellectual disability',
        'Social/Behavioural Issues',
        'Anxiety_disorder',
        'Jaundice',
        'Family_mem_with_ASD',
        'ASD_traits'
    ]

    for col in bool_columns:
        df[col] = df[col].replace({
            'Yes': 1,
            'No': 0
        })

    return df

def frequency_encoding(df):

    freq_ethnicity = df['Ethnicity'].value_counts(normalize=True)

    freq_test = df['Who_completed_the_test'].value_counts(normalize=True)

    df['Ethnicity_en'] = df['Ethnicity'].map(freq_ethnicity)

    df['Who_completed_the_test_en'] = df[
        'Who_completed_the_test'
    ].map(freq_test)

    df.drop('Ethnicity', axis=1, inplace=True)

    df.drop('Who_completed_the_test', axis=1, inplace=True)

    return df

def encode_sex(df):

    encoder = OrdinalEncoder()

    encoded = encoder.fit_transform(
        df["Sex"].values.reshape(-1, 1)
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=['Sex_en']
    )

    df = pd.concat([df, encoded_df], axis=1)

    df.drop('Sex', axis=1, inplace=True)

    return df, encoder

def drop_unused_columns(df):

    df.drop(df.columns[0], axis=1, inplace=True)

    return df

def preprocess_data(df):

    df = clean_data(df)

    df = handle_missing_values(df)

    df = create_result_feature(df)

    df = encode_yes_no(df)

    df = frequency_encoding(df)

    df, sex_encoder = encode_sex(df)

    df = drop_unused_columns(df)

    return df, sex_encoder

def train_model(df):

    X = df.drop('ASD_traits', axis=1)

    y = df['ASD_traits']

    X.columns = X.columns.astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=100,      
        learning_rate=0.1,     
        max_depth=5,          
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")

    print(f"F1 Score: {f1}")

    return model


def save_model(model, file_name='autism_model.pkl'):

    joblib.dump(model, file_name)

    print("Model Saved Successfully")

def load_model(file_name='autism_model.pkl'):

    model = joblib.load(file_name)

    return model

def preprocess_single_input(data_dict):

    df = pd.DataFrame([data_dict])

    df = clean_data(df)

    df = create_result_feature(df)

    df = encode_yes_no(df)

    df = frequency_encoding(df)

    df, _ = encode_sex(df)

    return df


def predict_autism(model, data_dict):

    processed_data = preprocess_single_input(data_dict)

    prediction = model.predict(processed_data)

    probability = model.predict_proba(processed_data)

    return {
        "prediction": int(prediction[0]),
        "probability": probability.tolist()
    }


if __name__ == "__main__":
    df = load_dataset(
        "autism_children_data.csv"
    )

    processed_df, sex_encoder = preprocess_data(df)
    model = train_model(processed_df)
    save_model(model)