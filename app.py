import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained model and training data
@st.cache(allow_output_mutation=True)
def load_model_and_data():
    model = joblib.load('spill_model.pkl')
    df = pd.read_excel('TW_SO_Sensitivity_Check).xlsx', sheet_name='Master', header=2)
    df['Spill reduction (%)'] = pd.to_numeric(df['Spill reduction (%)'], errors='coerce')
    # Define the feature names
    features = ['Number of upstream Nodes', 'Number of upstream  Pipes', 'Population Count', 'Modelled Peak DWF (l/s)',
                'Total model catchment (ha)', 'Contributing catchment (ha)', 'Total Impermeable (Roads,Roofs) (ha)',
                '% Impermeable total contributing', 'Storage Volume (m3)',
                'Type of catchment (Combined/Foul)', 'Soil Type', 'Model status']
    return model, df, features

model, df, features = load_model_and_data()

# Separate numeric and categorical
numeric_features = [f for f in features if df[f].dtype != 'object']
categorical_features = [f for f in features if df[f].dtype == 'object']

st.title('Spill Reduction Prediction Tool')
st.write('Provide catchment characteristics to predict the percentage spill reduction.')

# Create input form
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
    # Ensure string type and unique
    options = [str(x) for x in options]
    if '' not in options:
        options = [''] + options
    input_data[feature] = st.selectbox(feature, options)

# Predict button
if st.button('Predict'):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    # Use model to predict
    pred = model.predict(input_df)[0]
    # Confidence: compute std over trees
    # Extract preprocessor and regressor
    preproc = model.named_steps['preproc']
    regressor = model.named_steps['regressor']
    X_trans = preproc.transform(input_df)
    tree_preds = np.array([t.predict(X_trans) for t in regressor.estimators_])
    std = float(tree_preds.std())
    st.subheader('Results')
    st.write(f'Predicted spill reduction: {pred:.2f} %')
    st.write(f'Estimated uncertainty (std. dev.): {std:.2f} %')
