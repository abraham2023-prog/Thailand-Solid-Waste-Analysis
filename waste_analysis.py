# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os

# Build the absolute path to the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'SW_Thailand_2021_Labeled.csv')

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(csv_path)
        required_cols = [
            'Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste',
            'Pop', 'GPP_per_Capita', 'GPP_Industrial(%)', 'Visitors(ppl)',
            'GPP_Agriculture(%)', 'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {missing}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def train_models():
    try:
        df = load_data()
        if df is None:
            return None
            
        # Prepare features
        X = df[[
            'Pop', 'GPP_per_Capita', 'GPP_Industrial(%)', 'Visitors(ppl)',
            'GPP_Agriculture(%)', 'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)'
        ]]
        
        # Create binary classification targets (1 if above median)
        classification_targets = {}
        for waste_type in ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']:
            median_val = df[waste_type].median()
            classification_targets[waste_type] = (df[waste_type] > median_val).astype(int)
        
        models = {}
        for waste_type in ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']:
            # Regression (Random Forest)
            y_reg = df[waste_type]
            X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
            
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X_train, y_train)
            y_pred = rf_reg.predict(X_test)
            reg_r2 = r2_score(y_test, y_pred)
            reg_mse = mean_squared_error(y_test, y_pred)
            
            # Classification (Logistic Regression and Random Forest)
            y_clf = classification_targets[waste_type]
            X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
            
            # Scale data for Logistic Regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_clf)
            X_test_scaled = scaler.transform(X_test_clf)
            
            logreg = LogisticRegression(max_iter=1000)
            logreg.fit(X_train_scaled, y_train_clf)
            logreg_acc = accuracy_score(y_test_clf, logreg.predict(X_test_scaled))
            
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_clf.fit(X_train_clf, y_train_clf)
            rf_clf_acc = accuracy_score(y_test_clf, rf_clf.predict(X_test_clf))
            
            models[waste_type] = {
                'rf_regressor': rf_reg,
                'logreg_classifier': logreg,
                'rf_classifier': rf_clf,
                'scaler': scaler,
                'reg_r2': reg_r2,
                'reg_mse': reg_mse,
                'logreg_acc': logreg_acc,
                'rf_clf_acc': rf_clf_acc,
                'features': X.columns.tolist(),
                'median': df[waste_type].median()
            }
            
        return models

    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("🇹🇭 Thailand Waste Prediction System")
    st.write("Predicts waste generation using Random Forest and Logistic Regression")
    
    models = train_models()
    if models is None:
        st.stop()
    
    df = load_data()
    if df is None:
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Info", "Data Exploration"])
    
    with tab1:
        st.header("Waste Prediction")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pop = st.slider("Population", float(df['Pop'].min()), float(df['Pop'].max()), float(df['Pop'].median()))
                gpp_per_capita = st.slider("GPP per Capita", float(df['GPP_per_Capita'].min()), float(df['GPP_per_Capita'].max()), float(df['GPP_per_Capita'].median()))
                gpp_industrial = st.slider("Industrial GDP %", float(df['GPP_Industrial(%)'].min()), float(df['GPP_Industrial(%)'].max()), float(df['GPP_Industrial(%)'].median()))
                
            with col2:
                visitors = st.slider("Visitors", float(df['Visitors(ppl)'].min()), float(df['Visitors(ppl)'].max()), float(df['Visitors(ppl)'].median()))
                gpp_agri = st.slider("Agriculture GDP %", float(df['GPP_Agriculture(%)'].min()), float(df['GPP_Agriculture(%)'].max()), float(df['GPP_Agriculture(%)'].median()))
                gpp_services = st.slider("Services GDP %", float(df['GPP_Services(%)'].min()), float(df['GPP_Services(%)'].max()), float(df['GPP_Services(%)'].median()))
                
            with col3:
                age_0_5 = st.slider("Age 0-5 %", float(df['Age_0_5'].min()), float(df['Age_0_5'].max()), float(df['Age_0_5'].median()))
                msw_gen_rate = st.slider("MSW Gen Rate", float(df['MSW_GenRate(ton/d)'].min()), float(df['MSW_GenRate(ton/d)'].max()), float(df['MSW_GenRate(ton/d)'].median()))
            
            waste_type = st.selectbox("Select Waste Type", options=list(models.keys()))
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                input_data = pd.DataFrame([[
                    pop, gpp_per_capita, gpp_industrial, visitors,
                    gpp_agri, gpp_services, age_0_5, msw_gen_rate
                ]], columns=models[waste_type]['features'])
                
                # Regression prediction
                reg_pred = models[waste_type]['rf_regressor'].predict(input_data)[0]
                
                # Classification prediction
                scaled_input = models[waste_type]['scaler'].transform(input_data)
                logreg_pred = models[waste_type]['logreg_classifier'].predict(scaled_input)[0]
                rf_clf_pred = models[waste_type]['rf_classifier'].predict(input_data)[0]
                
                st.success("### Prediction Results")
                st.write(f"**Random Forest Regression Prediction:** {reg_pred:.2f} tons/day")
                st.write(f"**Logistic Regression Classification:** {'High' if logreg_pred == 1 else 'Low'} waste (threshold: {models[waste_type]['median']:.2f} tons/day)")
                st.write(f"**Random Forest Classification:** {'High' if rf_clf_pred == 1 else 'Low'} waste")
    
    with tab2:
        st.header("Model Information")
        
        waste_type = st.selectbox("Select Waste Type", options=list(models.keys()), key='model_select')
        model_info = models[waste_type]
        
        st.subheader("Random Forest Regressor")
        st.write(f"R-squared: {model_info['reg_r2']:.3f}")
        st.write(f"MSE: {model_info['reg_mse']:.3f}")
        
        st.subheader("Logistic Regression Classifier")
        st.write(f"Accuracy: {model_info['logreg_acc']:.3f}")
        
        st.subheader("Random Forest Classifier")
        st.write(f"Accuracy: {model_info['rf_clf_acc']:.3f}")
        
        st.subheader("Feature Importances (RF Regressor)")
        importances = pd.DataFrame({
            'Feature': model_info['features'],
            'Importance': model_info['rf_regressor'].feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importances.set_index('Feature'))
    
    with tab3:
        st.header("Data Exploration")
        st.write("### Descriptive Statistics")
        st.dataframe(df.describe())
        
        st.write("### Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numeric_cols].corr())

if __name__ == "__main__":
    main()
