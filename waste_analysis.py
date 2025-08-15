# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# XGBoost installation workaround
try:
    from xgboost import XGBRegressor
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==1.7.3"])
    from xgboost import XGBRegressor

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Thailand Waste Analysis Pro",
    page_icon="♻️",
    layout="wide"
)

# ----------------------------
# Data Loading and Preparation
# ----------------------------
@st.cache_data
def load_and_prepare_data():
    try:
        # Load data
        if not os.path.exists('SW_Thailand_2021_Labeled.csv'):
            st.error("Data file not found! Please ensure SW_Thailand_2021_Labeled.csv is in your app directory")
            return None
            
        df = pd.read_csv('SW_Thailand_2021_Labeled.csv')
        
        # Data Quality Report
        with st.expander("🔍 Initial Data Quality Report", expanded=True):
            st.write("### Missing Values Before Processing")
            missing_data = df.isnull().sum().to_frame("Missing Values")
            st.dataframe(missing_data.style.background_gradient(cmap='Reds'))
            
            # Remove completely empty columns
            empty_cols = df.columns[df.isnull().all()]
            if len(empty_cols) > 0:
                st.warning(f"Removing empty columns: {list(empty_cols)}")
                df = df.drop(columns=empty_cols)

        # Define targets and features
        waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        
        # Feature Engineering
        base_features = ['Pop', 'GPP_Industrial(%)', 'Visitors(ppl)', 
                       'GPP_Agriculture(%)', 'GPP_Services(%)', 'Age_0_5', 
                       'MSW_GenRate(ton/d)']
        
        # Create new features
        if 'Area' in df.columns:
            df['Population_Density'] = df['Pop'] / df['Area']
            base_features.append('Population_Density')
        
        df['Economic_Diversity'] = df[['GPP_Agriculture(%)', 
                                     'GPP_Industrial(%)', 
                                     'GPP_Services(%)']].std(axis=1)
        base_features.append('Economic_Diversity')
        
        # Add interaction terms
        df['Industrial_Service_Interaction'] = df['GPP_Industrial(%)'] * df['GPP_Services(%)']
        base_features.extend(['Industrial_Service_Interaction'])
        
        # Handle missing values
        features = [f for f in base_features if f in df.columns]
        
        # 1. Impute features using median
        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])
        
        # 2. For waste targets - use iterative imputation
        imputer = IterativeImputer(random_state=42, max_iter=10)
        df[waste_targets] = imputer.fit_transform(df[waste_targets])
        
        # Remove highly correlated features
        corr_matrix = df[features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        if to_drop:
            st.warning(f"🚀 Dropping highly correlated features: {to_drop}")
            features = [f for f in features if f not to_drop]
        
        # Final check
        with st.expander("✅ Final Data Quality Report"):
            st.write("Missing Values After Processing:")
            st.write(df.isnull().sum())
            st.write("\nDataset Shape:", df.shape)
            st.write("Features used:", features)
        
        return df[features + waste_targets]
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

# ----------------------------
# Feature Selection
# ----------------------------
def select_features(X, y):
    """
    Use Recursive Feature Elimination with Cross-Validation
    """
    estimator = RandomForestRegressor(random_state=42)
    selector = RFECV(estimator, step=1, cv=5, scoring='r2')
    selector = selector.fit(X, y)
    
    selected_features = X.columns[selector.support_]
    st.write(f"✅ Selected features: {list(selected_features)}")
    return selected_features

# ----------------------------
# Model Training with Hyperparameter Tuning
# ----------------------------
def train_optimized_model(X, y):
    """
    Train model with hyperparameter tuning
    """
    models = {
        'ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.01, 0.1, 1, 10, 100]}
        },
        'random_forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 3, 5, 10],
                'min_samples_leaf': [1, 5, 10]
            }
        }
    }
    
    best_models = {}
    for name, config in models.items():
        gs = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        gs.fit(X, y)
        best_models[name] = gs.best_estimator_
        st.write(f"🎯 Best {name} params:", gs.best_params_)
        st.write(f"Best {name} R²: {gs.best_score_:.3f}")
    
    return best_models

def train_ensemble_model(X, y):
    """
    Train an ensemble model combining multiple approaches
    """
    # Individual models
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
    # Ensemble
    ensemble = VotingRegressor(
        estimators=[('ridge', ridge), ('rf', rf), ('xgb', xgb)],
        weights=[1, 2, 2]  # Give more weight to tree-based models
    )
    
    # Cross-validate
    scores = cross_val_score(ensemble, X, y, cv=5, scoring='r2')
    st.write(f"🏆 Ensemble CV R²: {np.mean(scores):.3f}")
    
    # Fit final model
    ensemble.fit(X, y)
    return ensemble

# ----------------------------
# Visualization Functions
# ----------------------------
def plot_feature_importance(model, features, title):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        # For models without feature_importances_, use coefficients
        importance = np.abs(model.coef_)
    
    df = pd.DataFrame({'Feature': features, 'Importance': importance})
    df = df.sort_values('Importance', ascending=False)
    
    fig = px.bar(df, x='Feature', y='Importance',
                title=title, color='Importance')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals,
                    labels={'x': 'Predicted', 'y': 'Residuals'},
                    title=title,
                    trendline="lowess")
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig)

# ----------------------------
# Main Application
# ----------------------------
def main():
    st.title("🇹🇭 Thailand Solid Waste Prediction Pro")
    st.markdown("""
    Advanced waste generation prediction system with feature engineering and ensemble modeling
    """)
    
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🧪 Model Training", "📊 Predictions", "📈 Analysis"])
    
    with tab1:
        st.header("Model Training Configuration")
        target = st.selectbox("Select Target Variable", 
                            options=['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste'])
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                # Prepare data
                X = df.drop(columns=['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste'])
                y = df[target]
                
                # Feature selection
                selected_features = select_features(X, y)
                X = X[selected_features]
                
                # Train models
                optimized_models = train_optimized_model(X, y)
                ensemble = train_ensemble_model(X, y)
                
                # Store in session state
                st.session_state.models = {
                    'ridge': optimized_models['ridge'],
                    'rf': optimized_models['random_forest'],
                    'ensemble': ensemble,
                    'features': selected_features,
                    'target': target
                }
                
                # Evaluate on test set
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Make predictions
                y_pred = ensemble.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                st.success(f"🎉 Final Model Performance (Test Set)")
                col1, col2 = st.columns(2)
                col1.metric("R² Score", f"{r2:.3f}")
                col2.metric("MSE", f"{mse:.3f}")
                
                # Plot residuals
                plot_residuals(y_test, y_pred, "Ensemble Model Residuals")
    
    with tab2:
        st.header("Make Predictions")
        if 'models' not in st.session_state:
            st.warning("Please train models first in the Model Training tab")
            st.stop()
            
        model = st.session_state.models
        target = model['target']
        
        # Create input form
        st.subheader(f"Predict {target.replace('_', ' ')}")
        cols = st.columns(3)
        input_data = {}
        
        for i, feature in enumerate(model['features']):
            with cols[i % 3]:
                input_data[feature] = st.slider(
                    feature,
                    float(df[feature].min()),
                    float(df[feature].max()),
                    float(df[feature].median()),
                    help=f"Range: {df[feature].min():.2f} to {df[feature].max():.2f}"
                )
        
        if st.button("Predict", type="primary"):
            # Prepare input
            X_input = pd.DataFrame([input_data])
            
            # Make predictions
            ridge_pred = model['ridge'].predict(X_input)[0]
            rf_pred = model['rf'].predict(X_input)[0]
            ensemble_pred = model['ensemble'].predict(X_input)[0]
            
            # Display results
            st.success("### Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Ridge Regression", f"{ridge_pred:.2f} tons/day")
                st.metric("Random Forest", f"{rf_pred:.2f} tons/day")
                st.metric("Ensemble Model", f"{ensemble_pred:.2f} tons/day")
            
            with col2:
                st.write("### Model Confidence")
                st.write("The ensemble model combines predictions from:")
                st.write("- Ridge Regression (linear relationships)")
                st.write("- Random Forest (non-linear patterns)")
                st.write("- XGBoost (gradient boosting)")
    
    with tab3:
        st.header("Model Analysis")
        if 'models' not in st.session_state:
            st.warning("Please train models first in the Model Training tab")
            st.stop()
            
        model = st.session_state.models
        
        st.subheader("Feature Importance")
        plot_feature_importance(model['rf'], model['features'], 
                              "Random Forest Feature Importance")
        
        st.subheader("Model Comparison")
        models = {
            'Ridge': model['ridge'],
            'Random Forest': model['rf'],
            'Ensemble': model['ensemble']
        }
        
        # Cross-validate all models
        X = df[model['features']]
        y = df[model['target']]
        
        results = []
        for name, m in models.items():
            scores = cross_val_score(m, X, y, cv=5, scoring='r2')
            results.append({
                'Model': name,
                'Mean R²': np.mean(scores),
                'Std R²': np.std(scores)
            })
        
        results_df = pd.DataFrame(results)
        fig = px.bar(results_df, x='Model', y='Mean R²', error_y='Std R²',
                    title="Model Comparison (5-Fold CV)")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
