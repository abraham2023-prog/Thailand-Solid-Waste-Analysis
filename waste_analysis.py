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
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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
    
    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                df = df.drop(columns=[col])
                st.warning(f"Dropped non-numeric column: {col}")
    
    # Basic feature engineering
    if 'Area' in df.columns and 'Pop' in df.columns:
        df['Population_Density'] = df['Pop'] / df['Area']
    
    if all(col in df.columns for col in ['GPP_Agriculture(%)', 'GPP_Industrial(%)', 'GPP_Services(%)']):
        df['Economic_Diversity'] = df[['GPP_Agriculture(%)', 'GPP_Industrial(%)', 'GPP_Services(%)']].std(axis=1)
    
    # Handle missing values - two step process
    features = [col for col in df.columns if col not in waste_targets]
    
    # 1. Impute features using median
    imputer = SimpleImputer(strategy='median')
    df[features] = imputer.fit_transform(df[features])
    
    # 2. For waste targets - use iterative imputation
    imputer = IterativeImputer(random_state=42, max_iter=10)
    df[waste_targets] = imputer.fit_transform(df[waste_targets])
    
    return df

# ----------------------------
# Model Training
# ----------------------------
@st.cache_resource
def train_models(df):
    models = {}
    waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
    
    for target in waste_targets:
        if target not in df.columns:
            continue
            
        try:
            X = df.drop(columns=waste_targets)
            y = df[target]
            
            # Remove any remaining non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
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
                
                results[name] = {
                    'model': pipeline,
                    'r2': r2_score(y_test, pred),
                    'mse': mean_squared_error(y_test, pred),
                    'cv_r2': np.mean(cross_val_score(
                        pipeline, X, y, cv=5, scoring='r2'
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
                'features': X.columns.tolist()
            }
            
        except Exception as e:
            st.error(f"Error modeling {target}: {str(e)}")
    
    return models

# ----------------------------
# Visualization Functions
# ----------------------------
def plot_feature_importance(model, features, target):
    # Get feature importances based on model type
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

def plot_waste_distribution(df):
    waste_cols = [col for col in ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste'] 
                 if col in df.columns]
    fig = px.box(df[waste_cols], title="Waste Distribution Across Provinces")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Main Application
# ----------------------------
def main():
    st.title("🇹🇭 Thailand Waste Prediction System")
    st.markdown("Predicting waste generation patterns across Thailand's provinces")
    
    # Load data
    raw_df = load_and_prepare_data()
    if raw_df is None:
        st.stop()
    
    # Feature engineering
    df = enhanced_feature_engineering(raw_df)
    
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
            pred = models[target]['model'].predict(X_input)[0]
            
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
