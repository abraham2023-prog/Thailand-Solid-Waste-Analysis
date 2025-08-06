# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import os

# Build the absolute path to the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'SW_Thailand_2021_Labeled.csv')

# Cache data loading with small-data optimizations
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv(csv_path)
        
        # 1. Critical feature selection
        waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        base_features = ['Pop', 'GPP_per_Capita', 'GPP_Industrial(%)', 
                        'Visitors(ppl)', 'GPP_Agriculture(%)', 
                        'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)']
        
        # 2. Smart feature engineering
        if 'Area' in df.columns:
            df['Population_Density'] = df['Pop'] / df['Area']
            base_features.append('Population_Density')
        
        df['Economic_Diversity'] = df[['GPP_Agriculture(%)', 
                                     'GPP_Industrial(%)', 
                                     'GPP_Services(%)']].std(axis=1)
        base_features.append('Economic_Diversity')
        
        return df[base_features + waste_targets].dropna()
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

# Optimized modeling for all waste types
@st.cache_resource
def train_all_waste_models():
    try:
        df = load_and_prepare_data()
        if df is None:
            return None
            
        models = {}
        waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        
        for target in waste_targets:
            # Prepare features - exclude other waste types to prevent leakage
            other_targets = [wt for wt in waste_targets if wt != target]
            X = df.drop(columns=other_targets)
            y = df[target]
            
            # Robust Regression with LOOCV
            ridge = Ridge(alpha=1.0)
            loo_scores = cross_val_score(ridge, X, y, cv=LeaveOneOut(), scoring='r2')
            
            # Final model
            ridge.fit(X, y)
            
            # Balanced Classification
            median_val = y.median()
            y_clf = (y > median_val).astype(int)
            
            # Random Forest models
            rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_reg.fit(X, y)
            
            rf_clf = RandomForestClassifier(n_estimators=50, 
                                          class_weight='balanced',
                                          random_state=42)
            rf_clf.fit(X, y_clf)
            
            models[target] = {
                'ridge_regressor': ridge,
                'rf_regressor': rf_reg,
                'rf_classifier': rf_clf,
                'features': X.columns.tolist(),
                'median': median_val,
                'loo_r2': np.mean(loo_scores),
                'X': X,
                'y': y,
                'y_clf': y_clf
            }
            
        return models
    
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("🇹🇭 Comprehensive Waste Prediction System")
    
    models = train_all_waste_models()
    if models is None:
        st.stop()
    
    df = load_and_prepare_data()
    if df is None:
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["Predict", "Analyze", "Data"])
    
    with tab1:
        st.header("Make Predictions for All Waste Types")
        
        target = st.selectbox("Select Waste Type", options=list(models.keys()))
        model = models[target]
        
        cols = st.columns(3)
        input_data = {}
        
        # Dynamically create sliders for each feature
        for i, feature in enumerate(model['features']):
            if feature in df.columns:  # Only show features that exist in data
                with cols[i % 3]:
                    input_data[feature] = st.slider(
                        feature,
                        float(df[feature].min()),
                        float(df[feature].max()),
                        float(df[feature].median()),
                        help=f"Range: {df[feature].min():.2f} to {df[feature].max():.2f}"
                    )
        
        if st.button("Predict"):
            # Prepare input
            X_input = pd.DataFrame([input_data])
            
            # Regression predictions
            ridge_pred = model['ridge_regressor'].predict(X_input)[0]
            rf_pred = model['rf_regressor'].predict(X_input)[0]
            
            # Classification prediction
            clf_pred = model['rf_classifier'].predict(X_input)[0]
            clf_proba = model['rf_classifier'].predict_proba(X_input)[0]
            
            # Display results
            st.success(f"### {target.replace('_', ' ')} Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ridge Regression", f"{ridge_pred:.2f} tons/day")
                st.metric("Random Forest Regression", f"{rf_pred:.2f} tons/day")
                st.write(f"Median threshold: {model['median']:.2f} tons/day")
                
            with col2:
                st.metric("Classification", 
                         "High Waste" if clf_pred == 1 else "Low Waste",
                         f"Confidence: {max(clf_proba)*100:.1f}%")
                
                # Show class distribution
                class_dist = pd.DataFrame({
                    'Low Waste': [sum(model['y_clf'] == 0)],
                    'High Waste': [sum(model['y_clf'] == 1)]
                })
                st.bar_chart(class_dist.T)
    
    with tab2:
        st.header("Model Analysis for All Waste Types")
        
        target = st.selectbox("Select Waste Type", options=list(models.keys()), key='analysis_select')
        model = models[target]
        
        # Regression Analysis
        st.subheader("Regression Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Leave-One-Out R²", f"{model['loo_r2']:.3f}")
        with col2:
            y_pred = model['ridge_regressor'].predict(model['X'])
            st.metric("MSE", f"{mean_squared_error(model['y'], y_pred):.3f}")
        
        # Feature Importance
        st.write("### Feature Importance (Ridge Regression)")
        if hasattr(model['ridge_regressor'], 'coef_'):
            coefs = pd.DataFrame({
                'Feature': model['features'],
                'Coefficient': model['ridge_regressor'].coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            st.dataframe(coefs.style.format({'Coefficient': '{:.4f}'}))
        
        # Classification Analysis
        st.subheader("Classification Performance")
        y_clf_pred = model['rf_classifier'].predict(model['X'])
        st.write(f"Accuracy: {accuracy_score(model['y_clf'], y_clf_pred):.3f}")
        st.write("#### Classification Report:")
        st.text(classification_report(model['y_clf'], y_clf_pred))
    
    with tab3:
        st.header("Data Exploration")
        
        st.write("### All Waste Types Distribution")
        waste_cols = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        st.dataframe(df[waste_cols].describe().style.format("{:.2f}"))
        
        st.write("### Correlation Matrix (Waste Types)")
        st.dataframe(df[waste_cols].corr().style.background_gradient(cmap='Blues').format("{:.2f}"))
        
        st.write("### Download Full Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="all_waste_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
