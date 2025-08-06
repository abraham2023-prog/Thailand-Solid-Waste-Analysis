# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
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

# Optimized modeling without external dependencies
@st.cache_resource
def train_optimized_models():
    try:
        df = load_and_prepare_data()
        if df is None:
            return None
            
        models = {}
        
        for target in ['Food_Waste', 'Gen_Waste']:  # Focus on key targets
            X = df.drop(columns=['Recycl_Waste', 'Hazard_Waste'])  # Remove other targets
            if target in X.columns:
                X = X.drop(columns=[target])
            y = df[target]
            
            # Robust Regression with LOOCV
            ridge = Ridge(alpha=1.0)  # Default regularization
            loo_scores = cross_val_score(ridge, X, y, cv=LeaveOneOut(), scoring='r2')
            
            # Final model
            ridge.fit(X, y)
            
            # Balanced Classification
            median_val = y.median()
            y_clf = (y > median_val).astype(int)
            
            # Simple Random Forest for better small-data performance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            logreg = LogisticRegression(class_weight='balanced', max_iter=1000)
            logreg.fit(X, y_clf)
            
            models[target] = {
                'regressor': ridge,
                'rf_regressor': rf,
                'classifier': logreg,
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
    st.title("🇹🇭 Waste Prediction (Optimized for Small Data)")
    
    models = train_optimized_models()
    if models is None:
        st.stop()
    
    df = load_and_prepare_data()
    if df is None:
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["Predict", "Analyze", "Data"])
    
    with tab1:
        st.header("Make Predictions")
        
        target = st.selectbox("Select Waste Type", options=list(models.keys()))
        model = models[target]
        
        cols = st.columns(3)
        input_data = {}
        
        # Dynamically create sliders for each feature
        for i, feature in enumerate(model['features']):
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
            ridge_pred = model['regressor'].predict(X_input)[0]
            rf_pred = model['rf_regressor'].predict(X_input)[0]
            
            # Classification prediction
            clf_pred = model['classifier'].predict(X_input)[0]
            clf_proba = model['classifier'].predict_proba(X_input)[0]
            
            # Display results
            st.success("### Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ridge Regression", f"{ridge_pred:.2f} tons/day")
                st.metric("Random Forest", f"{rf_pred:.2f} tons/day")
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
        st.header("Model Analysis")
        
        target = st.selectbox("Select Waste Type", options=list(models.keys()), key='analysis_select')
        model = models[target]
        
        # Regression Analysis
        st.subheader("Regression Performance")
        st.write(f"Leave-One-Out R²: {model['loo_r2']:.3f}")
        
        # Feature Importance
        st.write("### Feature Importance (Ridge Regression)")
        if hasattr(model['regressor'], 'coef_'):
            coefs = pd.DataFrame({
                'Feature': model['features'],
                'Coefficient': model['regressor'].coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            st.dataframe(coefs.style.format({'Coefficient': '{:.4f}'}))
        
        # Classification Analysis
        st.subheader("Classification Performance")
        y_clf_pred = model['classifier'].predict(model['X'])
        st.write(f"Accuracy: {accuracy_score(model['y_clf'], y_clf_pred):.3f}")
        st.write("#### Classification Report:")
        st.text(classification_report(model['y_clf'], y_clf_pred))
    
    with tab3:
        st.header("Data Exploration")
        
        st.write("### Dataset Statistics")
        st.dataframe(df.describe().style.format("{:.2f}"))
        
        st.write("### Correlation Matrix")
        corr = df.corr()
        st.dataframe(corr.style.format("{:.2f}"))
        
        st.write("### Download Prepared Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="optimized_waste_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
