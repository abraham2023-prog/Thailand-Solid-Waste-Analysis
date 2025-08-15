# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import make_pipeline
import os

# ----------------------------
# Enhanced Feature Engineering
# ----------------------------
def enhanced_feature_engineering(df):
    # Original features
    waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
    base_features = ['Pop', 'GPP_per_Capita', 'GPP_Industrial(%)', 
                   'Visitors(ppl)', 'GPP_Agriculture(%)', 
                   'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)']
    
    # New feature: Population density
    if 'Area' in df.columns:
        df['Population_Density'] = df['Pop'] / df['Area']
        base_features.append('Population_Density')
    
    # Economic diversity index
    df['Economic_Diversity'] = df[['GPP_Agriculture(%)', 
                                 'GPP_Industrial(%)', 
                                 'GPP_Services(%)']].std(axis=1)
    base_features.append('Economic_Diversity')
    
    # Polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[['GPP_Industrial(%)', 'GPP_Services(%)', 'Pop']])
    poly_cols = ['Industrial_Service_Interaction', 'Industrial_Pop', 'Service_Pop']
    df[poly_cols] = poly_features[:, -3:]  # Take only interaction terms
    base_features.extend(poly_cols)
    
    # Log transformations
    for col in ['Pop', 'Visitors(ppl)', 'GPP_per_Capita']:
        df[f'log_{col}'] = np.log1p(df[col])
        base_features.append(f'log_{col}')
    
    return df, base_features, waste_targets

# ----------------------------
# Improved Model Training
# ----------------------------
@st.cache_resource
def train_enhanced_models(df):
    models = {}
    df, base_features, waste_targets = enhanced_feature_engineering(df)
    
    for target in waste_targets:
        try:
            X = df[base_features]
            y = df[target]
            
            # Feature selection
            selector = RFECV(RandomForestRegressor(n_estimators=50, random_state=42), 
                           step=1, cv=5, scoring='r2')
            selector.fit(X, y)
            selected_features = X.columns[selector.support_]
            X = X[selected_features]
            
            # Train-test split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Model pipelines with hyperparameter tuning
            models[target] = {
                'features': selected_features.tolist(),
                'scaler': RobustScaler(),
                'models': {
                    'ridge': GridSearchCV(
                        Ridge(),
                        {'alpha': np.logspace(-3, 3, 20)},
                        cv=5
                    ).fit(X_train, y_train),
                    
                    'rf': GridSearchCV(
                        RandomForestRegressor(random_state=42),
                        {'n_estimators': [50, 100],
                         'max_depth': [3, 5, None]},
                        cv=5
                    ).fit(X_train, y_train),
                    
                    'gboost': GridSearchCV(
                        GradientBoostingRegressor(random_state=42),
                        {'n_estimators': [50, 100],
                         'learning_rate': [0.01, 0.1]},
                        cv=5
                    ).fit(X_train, y_train)
                },
                'metrics': {
                    'cv_r2': np.mean(cross_val_score(
                        GradientBoostingRegressor(random_state=42),
                        X, y, cv=5, scoring='r2'
                    ))
                }
            }
            
            # Calculate validation metrics
            for name, model in models[target]['models'].items():
                pred = model.predict(X_val)
                models[target]['metrics'][f'{name}_r2'] = r2_score(y_val, pred)
                models[target]['metrics'][f'{name}_mse'] = mean_squared_error(y_val, pred)
                
        except Exception as e:
            st.error(f"Error modeling {target}: {str(e)}")
    
    return models

# ----------------------------
# Enhanced Visualization
# ----------------------------
def plot_enhanced_feature_importance(models, target):
    rf_model = models[target]['models']['rf'].best_estimator_
    features = models[target]['features']
    importance = rf_model.feature_importances_
    
    fig = px.bar(
        x=features,
        y=importance,
        title=f"Feature Importance for {target}",
        labels={'x': 'Features', 'y': 'Importance'},
        color=importance,
        color_continuous_scale='Bluered'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Main Application (Updated)
# ----------------------------
def main():
    st.title("🇹🇭 Enhanced Thailand Waste Prediction")
    
    # Load data
    df = load_and_prepare_data()  # Keep your existing loading function
    if df is None:
        st.stop()
    
    # Train enhanced models
    models = train_enhanced_models(df)
    
    # Prediction Interface
    st.header("Enhanced Predictions")
    target = st.selectbox("Select Waste Type", options=list(models.keys()))
    
    # Dynamic input sliders
    cols = st.columns(3)
    input_data = {}
    for i, feature in enumerate(models[target]['features']):
        with cols[i % 3]:
            input_data[feature] = st.slider(
                feature,
                float(df[feature].min()),
                float(df[feature].max()),
                float(df[feature].median()),
                help=f"Range: {df[feature].min():.2f} to {df[feature].max():.2f}"
            )
    
    if st.button("Predict", type="primary"):
        X_input = pd.DataFrame([input_data])
        results = {}
        
        for name, model in models[target]['models'].items():
            results[name] = model.predict(X_input)[0]
        
        # Display results
        st.success("## Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            for name, pred in results.items():
                st.metric(f"{name.upper()} Prediction", f"{pred:.2f} tons/day")
        
        with col2:
            st.write("### Model Performance")
            st.metric("Cross-Validated R²", 
                     f"{models[target]['metrics']['cv_r2']:.3f}")
            st.write("Validation R² Scores:")
            for name in models[target]['models'].keys():
                st.write(f"- {name}: {models[target]['metrics'][f'{name}_r2']:.3f}")

if __name__ == "__main__":
    main()
