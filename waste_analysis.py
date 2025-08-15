# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'SW_Thailand_2021_Labeled.csv')
        df = pd.read_csv(csv_path)
        
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

        return df
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

# ----------------------------
# Enhanced Feature Engineering
# ----------------------------
def enhanced_feature_engineering(df):
    waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
    
    # Economic features
    economic_features = ['GPP_Industrial(%)', 'GPP_Services(%)', 'GPP_Agriculture(%)']
    df['Economic_Activity'] = df[economic_features].mean(axis=1)
    df['Industry_Service_Ratio'] = df['GPP_Industrial(%)'] / (df['GPP_Services(%)'] + 1e-6)
    
    # Tourism features
    if 'Area' in df.columns:
        df['Tourism_Pressure'] = df['Visitors(ppl)'] / (df['Area'] + 1)
    else:
        df['Tourism_Pressure'] = df['Visitors(ppl)']
    
    # Demographic features
    df['Dependent_Ratio'] = df['Age_0_5'] / df['Pop']
    
    # Log transforms
    for col in ['Pop', 'Visitors(ppl)', 'MSW_GenRate(ton/d)']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    
    # Remove zero-variance features
    selector = VarianceThreshold(threshold=0.01)
    features = [col for col in df.columns if col not in waste_targets]
    df[features] = selector.fit_transform(df[features])
    features = df.columns[selector.get_support()]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[features] = imputer.fit_transform(df[features])
    
    # Impute targets
    imputer = IterativeImputer(random_state=42, max_iter=10)
    df[waste_targets] = imputer.fit_transform(df[waste_targets])
    
    return df

# ----------------------------
# Model Training with Diagnostics
# ----------------------------
@st.cache_resource
def train_models(df):
    models = {}
    waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
    
    for target in waste_targets:
        try:
            X = df.drop(columns=waste_targets)
            y = df[target]
            
            # Target transformation
            transform_target = st.checkbox(f"Transform {target} (recommended)", value=True, key=f"transform_{target}")
            if transform_target:
                qt = QuantileTransformer(output_distribution='normal', random_state=42)
                y_transformed = qt.fit_transform(y.values.reshape(-1, 1)).flatten()
            else:
                y_transformed = y.copy()
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_transformed, test_size=0.2, random_state=42
            )
            
            # Model pipelines
            pipelines = {
                'Ridge': make_pipeline(
                    RobustScaler(),
                    Ridge(alpha=1.0)
                ),
                'Random Forest': make_pipeline(
                    RobustScaler(),
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=5,
                        min_samples_leaf=5,
                        random_state=42
                    )
                ),
                'Gradient Boosting': make_pipeline(
                    RobustScaler(),
                    GradientBoostingRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42
                    )
                )
            }
            
            # Train and evaluate
            results = {}
            for name, pipeline in pipelines.items():
                pipeline.fit(X_train, y_train)
                pred = pipeline.predict(X_test)
                if transform_target:
                    pred = qt.inverse_transform(pred.reshape(-1, 1)).flatten()
                    y_test_actual = qt.inverse_transform(y_test.reshape(-1, 1)).flatten()
                else:
                    y_test_actual = y_test
                
                results[name] = {
                    'model': pipeline,
                    'r2': r2_score(y_test_actual, pred),
                    'mse': mean_squared_error(y_test_actual, pred),
                    'cv_r2': np.mean(cross_val_score(
                        pipeline, X, y_transformed, cv=5, scoring='r2'
                    ))
                }
            
            # Store best model
            best_model = max(results.items(), key=lambda x: x[1]['r2'])
            models[target] = {
                'model': best_model[1]['model'],
                'metrics': {
                    'test_r2': best_model[1]['r2'],
                    'test_mse': best_model[1]['mse'],
                    'cv_r2': best_model[1]['cv_r2']
                },
                'features': X.columns.tolist(),
                'transform': transform_target
            }
            
            # Plot diagnostics
            with st.expander(f"Diagnostics for {target}", expanded=False):
                plot_actual_vs_predicted(
                    y_test_actual, pred, 
                    f"Actual vs Predicted {target}"
                )
                plot_error_distribution(y_test_actual, pred)
                
        except Exception as e:
            st.error(f"Error modeling {target}: {str(e)}")
    
    return models

# ----------------------------
# Visualization Functions
# ----------------------------
def plot_feature_importance(model, features, target):
    if hasattr(model.named_steps.get('randomforestregressor', None), 'feature_importances_'):
        importance = model.named_steps['randomforestregressor'].feature_importances_
    elif hasattr(model.named_steps.get('gradientboostingregressor', None), 'feature_importances_'):
        importance = model.named_steps['gradientboostingregressor'].feature_importances_
    else:
        importance = np.abs(model.named_steps['ridge'].coef_)
    
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

def plot_actual_vs_predicted(y_true, y_pred, title):
    fig = px.scatter(
        x=y_true, y=y_pred,
        labels={'x': 'Actual', 'y': 'Predicted'},
        title=title,
        trendline="ols"
    )
    fig.add_shape(type="line", x0=min(y_true), y0=min(y_true),
                 x1=max(y_true), y1=max(y_true),
                 line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

def plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    fig = px.histogram(x=errors, nbins=30, title="Error Distribution")
    fig.add_vline(x=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

def plot_waste_distribution(df):
    waste_cols = [col for col in ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste'] 
                 if col in df.columns]
    fig = px.box(df[waste_cols], title="Waste Distribution Across Provinces")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Main Application
# ----------------------------
def main():
    st.title("🇹🇭 Thailand Waste Prediction System Pro")
    st.markdown("Advanced waste generation prediction with feature engineering and model diagnostics")
    
    # Load data
    raw_df = load_and_prepare_data()
    if raw_df is None:
        st.stop()
    
    # Feature engineering
    df = enhanced_feature_engineering(raw_df)
    
    # Show transformed data
    with st.expander("View Processed Data", expanded=False):
        st.dataframe(df.describe())
    
    # Train models
    models = train_models(df)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Analysis", "🗂️ Data Explorer"])
    
    with tab1:
        st.header("Waste Generation Predictions")
        target = st.selectbox("Select Waste Type", options=list(models.keys()))
        
        # Create input sliders
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
        
        if st.button("Predict Waste Generation", type="primary"):
            X_input = pd.DataFrame([input_data])
            pred = models[target]['model'].predict(X_input)
            
            if models[target]['transform']:
                # Create dummy qt for inverse transform
                qt = QuantileTransformer(output_distribution='normal', random_state=42)
                qt.fit(df[target].values.reshape(-1, 1))
                pred = qt.inverse_transform(pred.reshape(-1, 1)).flatten()[0]
            
            st.success(f"### Predicted {target.replace('_', ' ')}: {pred:.2f} tons/day")
            st.write(f"Model R² score: {models[target]['metrics']['test_r2']:.3f}")
    
    with tab2:
        st.header("Model Analysis")
        target = st.selectbox("Select Waste Type for Analysis", options=list(models.keys()), key='analysis')
        
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Test R² Score", f"{models[target]['metrics']['test_r2']:.3f}")
        col2.metric("Cross-Validated R²", f"{models[target]['metrics']['cv_r2']:.3f}")
        
        st.subheader("Feature Importance")
        plot_feature_importance(
            models[target]['model'],
            models[target]['features'],
            target
        )
    
    with tab3:
        st.header("Data Exploration")
        st.subheader("Waste Distribution")
        plot_waste_distribution(df)
        
        st.subheader("Correlation Matrix")
        corr = df.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            range_color=[-1, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
