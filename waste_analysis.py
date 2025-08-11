# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os

# Build the absolute path to the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'SW_Thailand_2021_Labeled.csv')

# Cache data loading with enhanced data quality checks
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv(csv_path)
        
        # Data Quality Report
        st.sidebar.write("### Data Quality Report")
        st.sidebar.write("Missing Values:", df.isnull().sum())
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            st.sidebar.warning(f"Constant columns detected: {constant_cols}")
        
        # 1. Critical feature selection
        waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        base_features = ['Pop', 'GPP_per_Capita', 'GPP_Industrial(%)', 
                        'Visitors(ppl)', 'GPP_Agriculture(%)', 
                        'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)']
        
        # Check target variable variance
        for target in waste_targets:
            if target in df.columns:
                if df[target].var() < 1e-6:
                    st.sidebar.error(f"Target {target} has near-zero variance!")
        
        # 2. Smart feature engineering
        if 'Area' in df.columns:
            df['Population_Density'] = df['Pop'] / df['Area']
            base_features.append('Population_Density')
        
        df['Economic_Diversity'] = df[['GPP_Agriculture(%)', 
                                     'GPP_Industrial(%)', 
                                     'GPP_Services(%)']].std(axis=1)
        base_features.append('Economic_Diversity')
        
        # Remove highly correlated features
        corr_matrix = df[base_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        if to_drop:
            st.sidebar.warning(f"Dropping highly correlated features: {to_drop}")
            base_features = [f for f in base_features if f not in to_drop]
        
        return df[base_features + waste_targets].dropna()
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

# Optimized modeling with regularization and proper validation
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
            
            # Feature scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            
            # Ridge Regression with LOOCV
            ridge = Ridge(alpha=1.0)
            try:
                loo_scores = cross_val_score(ridge, X_train, y_train, 
                                           cv=LeaveOneOut(), scoring='r2')
                loo_r2 = np.mean(loo_scores)
            except:
                loo_r2 = np.nan
                st.warning(f"LOOCV failed for {target}, possibly due to low variance")
            
            # Train final models
            ridge.fit(X_train, y_train)
            val_pred = ridge.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            
            # Lasso as alternative
            lasso = Lasso(alpha=0.1)
            lasso.fit(X_train, y_train)
            
            # Balanced Classification - using top quartile as threshold
            median_val = y.median()
            upper_quartile = y.quantile(0.75)
            y_clf_train = (y_train > upper_quartile).astype(int)
            y_clf_val = (y_val > upper_quartile).astype(int)
            
            # Random Forest models with regularization
            rf_reg = RandomForestRegressor(
                n_estimators=100, 
                max_depth=5,
                min_samples_leaf=5,
                random_state=42
            )
            rf_reg.fit(X_train, y_train)
            
            rf_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            )
            rf_clf.fit(X_train, y_clf_train)
            
            # Store all model information
            models[target] = {
                'ridge_regressor': ridge,
                'lasso_regressor': lasso,
                'rf_regressor': rf_reg,
                'rf_classifier': rf_clf,
                'features': X.columns.tolist(),
                'median': median_val,
                'upper_quartile': upper_quartile,
                'loo_r2': loo_r2,
                'val_r2': val_r2,
                'X_train': X_train,
                'y_train': y_train,
                'y_clf_train': y_clf_train,
                'X_val': X_val,
                'y_val': y_val,
                'y_clf_val': y_clf_val,
                'scaler': scaler
            }
            
        return models
    
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("🇹🇭 Robust Waste Prediction System")
    
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
            if feature in df.columns:
                with cols[i % 3]:
                    input_data[feature] = st.slider(
                        feature,
                        float(df[feature].min()),
                        float(df[feature].max()),
                        float(df[feature].median()),
                        help=f"Range: {df[feature].min():.2f} to {df[feature].max():.2f}"
                    )
        
        if st.button("Predict"):
            # Prepare and scale input
            X_input = pd.DataFrame([input_data])
            X_input_scaled = model['scaler'].transform(X_input)
            
            # Regression predictions
            ridge_pred = model['ridge_regressor'].predict(X_input_scaled)[0]
            lasso_pred = model['lasso_regressor'].predict(X_input_scaled)[0]
            rf_pred = model['rf_regressor'].predict(X_input_scaled)[0]
            
            # Classification prediction
            clf_pred = model['rf_classifier'].predict(X_input_scaled)[0]
            clf_proba = model['rf_classifier'].predict_proba(X_input_scaled)[0]
            
            # Display results
            st.success(f"### {target.replace('_', ' ')} Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ridge Regression", f"{ridge_pred:.2f} tons/day")
                st.metric("Lasso Regression", f"{lasso_pred:.2f} tons/day")
                st.metric("Random Forest Regression", f"{rf_pred:.2f} tons/day")
                st.write(f"Upper quartile threshold: {model['upper_quartile']:.2f} tons/day")
                
            with col2:
                st.metric("Classification", 
                         "High Waste" if clf_pred == 1 else "Low Waste",
                         f"Confidence: {max(clf_proba)*100:.1f}%")
                
                # Show class distribution
                class_dist = pd.DataFrame({
                    'Count': [sum(model['y_clf_train'] == 0), sum(model['y_clf_train'] == 1)],
                    'Waste Level': ['Low Waste', 'High Waste']
                })
                st.bar_chart(class_dist.set_index('Waste Level'))
    
    with tab2:
        st.header("Model Analysis for All Waste Types")
        
        target = st.selectbox("Select Waste Type", options=list(models.keys()), key='analysis_select')
        model = models[target]
        
        # Regression Analysis
        st.subheader("Regression Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("LOO CV R²", f"{model['loo_r2']:.3f}")
        with col2:
            st.metric("Validation R²", f"{model['val_r2']:.3f}")
        with col3:
            y_pred = model['ridge_regressor'].predict(model['X_val'])
            st.metric("Validation MSE", f"{mean_squared_error(model['y_val'], y_pred):.3f}")
        
        # Feature Importance
        st.write("### Feature Importance")
        tab_coef, tab_imp = st.tabs(["Regression Coefficients", "RF Feature Importance"])
        
        with tab_coef:
            if hasattr(model['ridge_regressor'], 'coef_'):
                coefs = pd.DataFrame({
                    'Feature': model['features'],
                    'Ridge Coef': model['ridge_regressor'].coef_,
                    'Lasso Coef': model['lasso_regressor'].coef_
                }).sort_values('Ridge Coef', key=abs, ascending=False)
                st.dataframe(coefs.style.format({'Ridge Coef': '{:.4f}', 'Lasso Coef': '{:.4f}'}))
        
        with tab_imp:
            if hasattr(model['rf_regressor'], 'feature_importances_'):
                importances = pd.DataFrame({
                    'Feature': model['features'],
                    'Importance': model['rf_regressor'].feature_importances_
                }).sort_values('Importance', ascending=False)
                st.dataframe(importances.style.format({'Importance': '{:.4f}'}))
                st.bar_chart(importances.set_index('Feature'))
        
        # Classification Analysis
        st.subheader("Classification Performance")
        y_clf_pred = model['rf_classifier'].predict(model['X_val'])
        st.write(f"Validation Accuracy: {accuracy_score(model['y_clf_val'], y_clf_pred):.3f}")
        st.write("#### Classification Report:")
        st.text(classification_report(model['y_clf_val'], y_clf_pred))
        
        # Actual vs Predicted plot
        st.write("### Actual vs Predicted Values")
        y_pred = model['ridge_regressor'].predict(model['X_val'])
        plot_data = pd.DataFrame({
            'Actual': model['y_val'],
            'Predicted': y_pred
        })
        st.line_chart(plot_data)
    
    with tab3:
        st.header("Data Exploration")
        
        st.write("### All Waste Types Distribution")
        waste_cols = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        st.dataframe(df[waste_cols].describe().style.format("{:.2f}"))
        
        st.write("### Correlation Between Waste Types")
        st.table(df[waste_cols].corr().round(2))
        
        st.write("### Feature Distributions")
        selected_feature = st.selectbox("Select feature to visualize", options=model['features'])
        st.bar_chart(df[selected_feature])
        
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
