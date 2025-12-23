import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def load_and_train():
    # Load Excel file
    df = pd.read_excel('TW_SO_Sensitivity_Check).xlsx', sheet_name='Master', header=2)
    df['Spill reduction (%)'] = pd.to_numeric(df['Spill reduction (%)'], errors='coerce')

    # Define predictors
    features = [
        'Number of upstream Nodes',
        'Number of upstream  Pipes',
        'Population Count',
        'Modelled Peak DWF (l/s)',
        'Total model catchment (ha)',
        'Contributing catchment (ha)',
        'Total Impermeable (Roads,Roofs) (ha)',
        '% Impermeable total contributing',
        'Storage Volume (m3)',
        'Type of catchment (Combined/Foul)',
        'Soil Type',
        'Model status'
    ]
    features = [f for f in features if f in df.columns]

    numeric_features = [f for f in features if df[f].dtype != 'object']
    categorical_features = [f for f in features if df[f].dtype == 'object']

    mask = df['Spill reduction (%)'].notnull()
    X = df.loc[mask, features]
    y = df.loc[mask, 'Spill reduction (%)']

    # Preprocessing
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ('preproc', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=0))
    ])
    model.fit(X, y)

    return df, features, numeric_features, categorical_features, model

df, features, numeric_features, categorical_features, model = load_and_train()

st.title('Spill Reduction Prediction Tool')
st.write('Provide catchment characteristics to predict the percentage spill reduction.')

input_data = {}

st.header('Numeric inputs')
for feature in numeric_features:
    min_val = float(df[feature].min(skipna=True)) if not pd.isna(df[feature].min(skipna=True)) else 0.0
    max_val = float(df[feature].max(skipna=True)) if not pd.isna(df[feature].max(skipna=True)) else 1.0
    mean_val = float(df[feature].median(skipna=True)) if not pd.isna(df[feature].median(skipna=True)) else 0.0
    input_data[feature] = st.number_input(feature, value=mean_val, min_value=min_val, max_value=max_val)

st.header('Categorical inputs')
for feature in categorical_features:
    options = df[feature].dropna().unique().tolist()
    options = [str(x) for x in options]
    if '' not in options:
        options = [''] + options
    input_data[feature] = st.selectbox(feature, options)

if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]

    estimator = model.named_steps['regressor']
    preproc = model.named_steps['preproc']
    X_trans = preproc.transform(input_df)
    tree_preds = np.array([t.predict(X_trans) for t in estimator.estimators_])
    std = float(tree_preds.std())

    st.subheader('Results')
    st.write(f'Predicted spill reduction: {pred:.2f} %')
    st.write(f'Estimated uncertainty (std. dev.): {std:.2f} %')
