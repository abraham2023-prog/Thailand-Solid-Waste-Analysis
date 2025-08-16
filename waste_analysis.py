# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
import os

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
        
        # Columns to remove
        cols_to_drop = [
            'Prov', 'Year_Thai', 'Pop', 'Age_0_5','Age_6_17', 'Age_18_24', 'Age_25_44', 'Age_45_64','Age_65plus', 
            'SAO', 'MSW_GenRate(kg/c/d)', 'Area_km2', 'Employed', 'Unemployed',
            'LAO_Special', 'City_Muni', 'MSW_Reclycled', 
            'Town_Muni', 'Subdist_Muni', 'District_BKK', 'Year', 'Pop_Density'
        ]
        
        # Remove specified columns if they exist
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
            st.warning(f"Removed specified columns: {existing_cols_to_drop}")
        
        # Data Quality Report
        with st.expander("🔍 Initial Data Quality Report", expanded=True):
            st.write("### Missing Values Before Processing")
            missing_data = df.isnull().sum().to_frame("Missing Values")
            st.dataframe(missing_data.style.background_gradient(cmap='Reds'))
            
            # Remove non-numeric columns
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns
            if len(non_numeric_cols) > 0:
                st.warning(f"Removing non-numeric columns: {list(non_numeric_cols)}")
                df = df.drop(columns=non_numeric_cols)
            
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
# Feature Engineering with Correlation-Based Multicollinearity Handling
# ----------------------------
def enhanced_feature_engineering(df):
    waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
    
    # Basic feature engineering
    if 'Area' in df.columns and 'Pop' not in df.columns:  # Since we're removing Pop
        # If you have another population column you want to use, change this
        st.warning("Population column not available for density calculation")
    elif 'Area' in df.columns:
        df['Population_Density'] = df['Pop'] / (df['Area'] + 1e-6)
    
    if all(col in df.columns for col in ['GPP_Agriculture(%)', 'GPP_Industrial(%)', 'GPP_Services(%)']):
        df['Economic_Balance'] = (df['GPP_Industrial(%)'] + 1e-6) / (df['GPP_Services(%)'] + 1e-6)
    
    # Handle multicollinearity using correlation analysis
    features = [col for col in df.columns if col not in waste_targets]
    corr_matrix = df[features].corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.8
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    if len(to_drop) > 0:
        st.warning(f"Removing highly correlated features: {to_drop}")
        features = [f for f in features if f not in to_drop]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[features] = imputer.fit_transform(df[features])
    
    # Impute targets
    imputer = IterativeImputer(random_state=42, max_iter=10)
    df[waste_targets] = imputer.fit_transform(df[waste_targets])
    
    return df[features + waste_targets]

# ----------------------------
# Model Training
# ----------------------------
@st.cache_resource
def train_models(df):
    models = {}
    waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for target in waste_targets:
        if target not in df.columns:
            continue
            
        try:
            X = df.drop(columns=waste_targets)
            y = df[target]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Model pipelines - Added 3 new models
            pipelines = {
                'Ridge': make_pipeline(
                    RobustScaler(),
                    Ridge(alpha=10.0)
                ),
                'Random Forest': make_pipeline(
                    RobustScaler(),
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=3,
                        min_samples_leaf=10,
                        random_state=42
                    )
                ),
                'Gradient Boosting': make_pipeline(
                    RobustScaler(),
                    GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42
                    )
                ),
                'SVR': make_pipeline(
                    RobustScaler(),
                    SVR(kernel='rbf', C=1.0, epsilon=0.1)
                ),
                'ElasticNet': make_pipeline(
                    RobustScaler(),
                    ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                )
            }
            
            # Train and evaluate
            results = {}
            for name, pipeline in pipelines.items():
                pipeline.fit(X_train, y_train)
                pred = pipeline.predict(X_test)
                
                cv_scores = cross_val_score(
                    pipeline, X, y, cv=cv, scoring='r2'
                )
                
                results[name] = {
                    'model': pipeline,
                    'test_r2': r2_score(y_test, pred),
                    'test_mse': mean_squared_error(y_test, pred),
                    'cv_r2_mean': np.mean(cv_scores),
                    'cv_r2_std': np.std(cv_scores),
                    'features': X.columns.tolist()
                }
            
            models[target] = results
            
        except Exception as e:
            st.error(f"Error modeling {target}: {str(e)}")
    
    return models

# ----------------------------
# Visualization Functions
# ----------------------------
def plot_feature_importance(model, features, model_name):
    if 'randomforestregressor' in model.named_steps:
        importance = model.named_steps['randomforestregressor'].feature_importances_
    else:
        importance = np.abs(model.named_steps['ridge'].coef_)
    
    fig = px.bar(
        x=features,
        y=importance,
        title=f"Feature Importance ({model_name})",
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
    st.title("🇹🇭 Thailand Waste Prediction System Pro")
    st.markdown("Optimized waste generation prediction with correlation-based feature selection")
    
    # Load data
    raw_df = load_and_prepare_data()
    if raw_df is None:
        st.stop()
    
    # Feature engineering
    df = enhanced_feature_engineering(raw_df)
    
    # Show processed data
    with st.expander("View Processed Data", expanded=False):
        st.write("### Missing Values After Processing")
        st.write(df.isnull().sum())
        st.write("### Dataset Shape:", df.shape)
    
    # Train models
    models = train_models(df)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Analysis", "🗂️ Data Explorer"])
    
    with tab1:
        st.header("Waste Generation Predictions")
        target = st.selectbox("Select Waste Type", options=list(models.keys()))
        
        model_choice = st.selectbox(
            "Select Model",
            options=list(models[target].keys()),
            format_func=lambda x: f"{x} (Test R²: {models[target][x]['test_r2']:.3f})"
        )
        
        # Create input sliders
        cols = st.columns(3)
        input_data = {}
        features = models[target][model_choice]['features']
        for i, feature in enumerate(features):
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
            model = models[target][model_choice]['model']
            pred = model.predict(X_input)[0]
            
            st.success(f"### Predicted {target.replace('_', ' ')}: {pred:.2f} tons/day")
            st.write(f"Model: {model_choice}")
            st.write(f"Test R²: {models[target][model_choice]['test_r2']:.3f}")
            st.write(f"CV R²: {models[target][model_choice]['cv_r2_mean']:.3f} ± {models[target][model_choice]['cv_r2_std']:.3f}")
    
    with tab2:
        st.header("Model Analysis")
        target = st.selectbox("Select Waste Type for Analysis", options=list(models.keys()), key='analysis')
        
        st.subheader("Feature Importance")
        model_choice = st.selectbox(
            "Select Model for Feature Importance",
            options=list(models[target].keys()),
            key='feature_importance'
        )
        plot_feature_importance(
            models[target][model_choice]['model'],
            models[target][model_choice]['features'],
            model_choice
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
