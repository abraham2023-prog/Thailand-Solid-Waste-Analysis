# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import LeaveOneOut, train_test_split
from skopt import BayesSearchCV
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import os
import matplotlib.pyplot as plt

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

# Bayesian-optimized modeling
@st.cache_resource
def train_bayesian_models():
    try:
        df = load_and_prepare_data()
        if df is None:
            return None
            
        models = {}
        loo = LeaveOneOut()
        
        for target in ['Food_Waste', 'Gen_Waste']:  # Focus on key targets
            X = df.drop(columns=['Recycl_Waste', 'Hazard_Waste'])  # Remove other targets
            if target in X.columns:
                X = X.drop(columns=[target])
            y = df[target]
            
            # Bayesian Ridge Regression
            ridge_opt = BayesSearchCV(
                Ridge(),
                {
                    'alpha': (1e-6, 1e+6, 'log-uniform')
                },
                n_iter=32,
                cv=loo,
                random_state=42
            )
            ridge_opt.fit(X, y)
            
            # Balanced Classification
            median_val = y.median()
            y_clf = (y > median_val).astype(int)
            
            bbc = BalancedBaggingClassifier(
                estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
                n_estimators=20,
                random_state=42,
                sampling_strategy='auto'
            )
            bbc.fit(X, y_clf)
            
            models[target] = {
                'regressor': ridge_opt.best_estimator_,
                'classifier': bbc,
                'features': X.columns.tolist(),
                'median': median_val,
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
    st.title("🇹🇭 Small-Data Waste Prediction System")
    
    models = train_bayesian_models()
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
            
            # Regression prediction
            reg_pred = model['regressor'].predict(X_input)[0]
            
            # Classification prediction
            clf_pred = model['classifier'].predict(X_input)[0]
            clf_proba = model['classifier'].predict_proba(X_input)[0]
            
            # Display results
            st.success("### Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Regression Prediction", f"{reg_pred:.2f} tons/day")
                st.write(f"Median threshold: {model['median']:.2f} tons/day")
                
                # Uncertainty estimation
                if hasattr(model['regressor'], 'alpha'):
                    st.write(f"Model regularization (alpha): {model['regressor'].alpha:.4f}")
                
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
        st.subheader("Bayesian Ridge Regression")
        y_pred = model['regressor'].predict(model['X'])
        st.write(f"LOO Cross-Validated R²: {r2_score(model['y'], y_pred):.3f}")
        st.write(f"Mean Squared Error: {mean_squared_error(model['y'], y_pred):.3f}")
        
        # Partial Dependence Plot
        st.write("### Partial Dependence (Top 2 Features)")
        fig, ax = plt.subplots(figsize=(10, 4))
        PartialDependenceDisplay.from_estimator(
            model['regressor'],
            model['X'],
            features=[0, 1],
            ax=ax
        )
        st.pyplot(fig)
        
        # Classification Analysis
        st.subheader("Balanced Classification")
        y_clf_pred = model['classifier'].predict(model['X'])
        st.write(f"Accuracy: {accuracy_score(model['y_clf'], y_clf_pred):.3f}")
        st.write("#### Classification Report:")
        st.text(classification_report(model['y_clf'], y_clf_pred))
        
        # Feature Importance
        st.write("### Feature Importance")
        if hasattr(model['regressor'], 'coef_'):
            coefs = pd.DataFrame({
                'Feature': model['features'],
                'Coefficient': model['regressor'].coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            st.dataframe(coefs.style.format({'Coefficient': '{:.4f}'}))
    
    with tab3:
        st.header("Data Exploration")
        
        st.write("### Dataset Statistics")
        st.dataframe(df.describe().style.format("{:.2f}"))
        
        st.write("### Correlation Matrix")
        corr = df.corr()
        st.dataframe(corr.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.2f}"))
        
        st.write("### Download Prepared Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="optimized_waste_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    plt.style.use('ggplot')
    main()
