pip install matplotlib
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import os
import matplotlib.pyplot as plt

# Build the absolute path to the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'SW_Thailand_2021_Labeled.csv')

# Enhanced data loading with thorough diagnostics
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv(csv_path)
        
        # Data Quality Report
        with st.expander("Data Quality Report"):
            st.write("### Missing Values")
            st.write(df.isnull().sum())
            
            st.write("### Basic Statistics")
            st.write(df.describe().T)
            
            # Check for constant columns
            constant_cols = [col for col in df.columns if df[col].nunique() == 1]
            if constant_cols:
                st.warning(f"Constant columns detected: {constant_cols}")
                df = df.drop(columns=constant_cols)
        
        # Critical feature selection
        waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        base_features = ['Pop', 'GPP_per_Capita', 'GPP_Industrial(%)', 
                       'Visitors(ppl)', 'GPP_Agriculture(%)', 
                       'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)']
        
        # Check target variable variance
        for target in waste_targets:
            if target in df.columns:
                var = df[target].var()
                st.write(f"Variance of {target}: {var:.4f}")
                if var < 1e-6:
                    st.error(f"Target {target} has near-zero variance!")
                    # Apply log transformation if values are positive
                    if (df[target] > 0).all():
                        df[target] = np.log1p(df[target])
                        st.warning(f"Applied log transformation to {target}")
        
        # Feature engineering
        if 'Area' in df.columns:
            df['Population_Density'] = df['Pop'] / df['Area']
            base_features.append('Population_Density')
        
        df['Economic_Diversity'] = df[['GPP_Agriculture(%)', 
                                     'GPP_Industrial(%)', 
                                     'GPP_Services(%)']].std(axis=1)
        base_features.append('Economic_Diversity')
        
        # Remove highly correlated features
        corr_matrix = df[base_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        if to_drop:
            st.warning(f"Dropping highly correlated features: {to_drop}")
            base_features = [f for f in base_features if f not in to_drop]
        
        # Final feature selection
        features = [f for f in base_features if f in df.columns]
        final_df = df[features + waste_targets].dropna()
        
        # Remove low-variance features
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(final_df[features])
        selected_features = [f for f, s in zip(features, selector.get_support()) if s]
        if len(selected_features) < len(features):
            st.warning(f"Removed low-variance features: {set(features) - set(selected_features)}")
            features = selected_features
        
        return final_df[features + waste_targets]
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

# Robust modeling with extensive diagnostics
@st.cache_resource
def train_all_waste_models():
    try:
        df = load_and_prepare_data()
        if df is None:
            return None
            
        models = {}
        waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        
        for target in waste_targets:
            if target not in df.columns:
                continue
                
            # Prepare features - exclude other waste types to prevent leakage
            other_targets = [wt for wt in waste_targets if wt != target]
            X = df.drop(columns=other_targets)
            y = df[target]
            
            # Check for sufficient variance
            if y.var() < 1e-6:
                st.warning(f"Skipping {target} due to insufficient variance")
                continue
            
            # Feature scaling with RobustScaler
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            
            # Ridge Regression with diagnostics
            ridge = Ridge(alpha=1.0)
            try:
                loo_scores = cross_val_score(ridge, X_train, y_train, 
                                          cv=LeaveOneOut(), scoring='r2')
                loo_r2 = np.mean(loo_scores)
            except Exception as e:
                st.warning(f"LOOCV failed for {target}: {str(e)}")
                loo_r2 = np.nan
            
            ridge.fit(X_train, y_train)
            ridge_pred = ridge.predict(X_val)
            ridge_r2 = r2_score(y_val, ridge_pred)
            ridge_mse = mean_squared_error(y_val, ridge_pred)
            
            # Check for suspiciously perfect fit
            if ridge_r2 > 0.999:
                st.warning(f"Suspiciously perfect R² ({ridge_r2:.3f}) for {target}")
                ridge_r2 = np.nan
                ridge_mse = np.nan
            
            # Lasso Regression
            lasso = Lasso(alpha=0.01, max_iter=10000)
            lasso.fit(X_train, y_train)
            
            # Random Forest Regressor with stricter parameters
            rf_reg = RandomForestRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42
            )
            rf_reg.fit(X_train, y_train)
            
            # Classification setup - using meaningful threshold
            threshold = y.quantile(0.75)  # Top 25% as high waste
            y_clf = (y > threshold).astype(int)
            
            # Re-split for classification
            X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
                X_scaled, y_clf, test_size=0.3, random_state=42
            )
            
            # Random Forest Classifier
            rf_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42
            )
            rf_clf.fit(X_train_clf, y_train_clf)
            
            # Store all model information
            models[target] = {
                'ridge_regressor': ridge,
                'lasso_regressor': lasso,
                'rf_regressor': rf_reg,
                'rf_classifier': rf_clf,
                'features': X.columns.tolist(),
                'threshold': threshold,
                'loo_r2': loo_r2,
                'val_r2': ridge_r2,
                'val_mse': ridge_mse,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_train_clf': X_train_clf,
                'y_train_clf': y_train_clf,
                'X_val_clf': X_val_clf,
                'y_val_clf': y_val_clf,
                'scaler': scaler
            }
            
        return models
    
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide")
    st.title("🇹🇭 Robust Waste Prediction System with Diagnostics")
    
    models = train_all_waste_models()
    if models is None:
        st.stop()
    
    df = load_and_prepare_data()
    if df is None:
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["Predict", "Analyze", "Data Diagnostics"])
    
    with tab1:
        st.header("Make Predictions")
        
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
                st.metric("Ridge Regression", f"{ridge_pred:.4f} tons/day")
                st.metric("Lasso Regression", f"{lasso_pred:.4f} tons/day")
                st.metric("Random Forest Regression", f"{rf_pred:.4f} tons/day")
                st.write(f"Classification threshold: {model['threshold']:.4f} tons/day")
                
            with col2:
                st.metric("Classification", 
                         "High Waste" if clf_pred == 1 else "Low Waste",
                         f"Confidence: {max(clf_proba)*100:.1f}%")
                
                # Show class distribution
                class_dist = pd.DataFrame({
                    'Count': [sum(model['y_train_clf'] == 0), sum(model['y_train_clf'] == 1)],
                    'Waste Level': ['Low Waste', 'High Waste']
                })
                st.bar_chart(class_dist.set_index('Waste Level'))
    
    with tab2:
        st.header("Model Analysis")
        
        target = st.selectbox("Select Waste Type", options=list(models.keys()), key='analysis_select')
        model = models[target]
        
        # Regression Analysis
        st.subheader("Regression Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("LOO CV R²", f"{model['loo_r2']:.4f}" if not np.isnan(model['loo_r2']) else "N/A")
        with col2:
            st.metric("Validation R²", f"{model['val_r2']:.4f}" if not np.isnan(model['val_r2']) else "N/A")
        with col3:
            st.metric("Validation MSE", f"{model['val_mse']:.6f}" if not np.isnan(model['val_mse']) else "N/A")
        
        # Feature Importance
        st.subheader("Feature Analysis")
        tab_coef, tab_imp, tab_data = st.tabs(["Coefficients", "Importance", "Data"])
        
        with tab_coef:
            if hasattr(model['ridge_regressor'], 'coef_'):
                coefs = pd.DataFrame({
                    'Feature': model['features'],
                    'Ridge': model['ridge_regressor'].coef_,
                    'Lasso': model['lasso_regressor'].coef_
                }).sort_values('Ridge', key=abs, ascending=False)
                
                # Highlight suspicious coefficients
                def highlight_suspicious(row):
                    if abs(row['Ridge']) > 1e6 or abs(row['Lasso']) > 1e6:
                        return ['background-color: yellow']*3
                    elif abs(row['Ridge']) < 1e-6 and abs(row['Lasso']) < 1e-6:
                        return ['background-color: lightgray']*3
                    else:
                        return ['']*3
                
                st.dataframe(coefs.style.apply(highlight_suspicious, axis=1).format({
                    'Ridge': '{:.6f}',
                    'Lasso': '{:.6f}'
                }))
        
        with tab_imp:
            if hasattr(model['rf_regressor'], 'feature_importances_'):
                importances = pd.DataFrame({
                    'Feature': model['features'],
                    'Importance': model['rf_regressor'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots()
                ax.barh(importances['Feature'], importances['Importance'])
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importances')
                st.pyplot(fig)
                
                st.dataframe(importances.style.format({'Importance': '{:.6f}'}))
        
        with tab_data:
            st.write("### Target Variable Distribution")
            fig, ax = plt.subplots()
            ax.hist(model['y_train'], bins=30)
            ax.axvline(model['threshold'], color='r', linestyle='--', label='Threshold')
            ax.set_xlabel(target)
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)
            
            st.write(f"Threshold value: {model['threshold']:.6f}")
            st.write(f"Mean: {np.mean(model['y_train']):.6f}")
            st.write(f"Std: {np.std(model['y_train']):.6f}")
        
        # Classification Analysis
        st.subheader("Classification Performance")
        y_clf_pred = model['rf_classifier'].predict(model['X_val_clf'])
        st.write(f"Validation Accuracy: {accuracy_score(model['y_val_clf'], y_clf_pred):.4f}")
        
        st.write("#### Classification Report:")
        st.text(classification_report(model['y_val_clf'], y_clf_pred))
        
        # Plot predictions
        st.write("### Validation Set Predictions")
        fig, ax = plt.subplots()
        ax.scatter(model['y_val'], model['ridge_regressor'].predict(model['X_val']))
        ax.plot([min(model['y_val']), max(model['y_val'])], 
               [min(model['y_val']), max(model['y_val'])], 
               'r--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        st.pyplot(fig)
    
    with tab3:
        st.header("Data Diagnostics")
        
        st.write("### Target Variables Distribution")
        waste_cols = [col for col in ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste'] 
                     if col in df.columns]
        
        fig, axes = plt.subplots(len(waste_cols), 1, figsize=(10, 3*len(waste_cols)))
        for ax, col in zip(axes, waste_cols):
            ax.hist(df[col], bins=30)
            ax.set_title(col)
        st.pyplot(fig)
        
        st.write("### Feature-Target Relationships")
        selected_feature = st.selectbox("Select feature", options=model['features'])
        selected_target = st.selectbox("Select target", options=waste_cols)
        
        fig, ax = plt.subplots()
        ax.scatter(df[selected_feature], df[selected_target])
        ax.set_xlabel(selected_feature)
        ax.set_ylabel(selected_target)
        st.pyplot(fig)
        
        st.write("### Correlation Matrix")
        corr = df[model['features'] + waste_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
