# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Thailand Waste Analysis",
    page_icon="🗑️",
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

        # Define targets and features
        waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        base_features = ['Pop', 'GPP_Industrial(%)', 
                       'Visitors(ppl)', 'GPP_Agriculture(%)', 
                       'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)']
        
        # Feature Engineering
        if 'Area' in df.columns:
            df['Population_Density'] = df['Pop'] / df['Area']
            base_features.append('Population_Density')
        
        df['Economic_Diversity'] = df[['GPP_Agriculture(%)', 
                                     'GPP_Industrial(%)', 
                                     'GPP_Services(%)']].std(axis=1)
        base_features.append('Economic_Diversity')
        
        # Handle missing values - two step process
        features = [f for f in base_features if f in df.columns]
        
        # 1. Impute features using median
        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])
        
        # 2. For waste targets - use iterative imputation
        imputer = IterativeImputer(random_state=42, max_iter=10)
        df[waste_targets] = imputer.fit_transform(df[waste_targets])
        
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
# Model Training
# ----------------------------
@st.cache_resource
def train_all_waste_models(df):
    try:
        models = {}
        waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        
        for target in waste_targets:
            if target not in df.columns:
                continue
                
            # Prepare data for this target
            X = df.drop(columns=waste_targets)
            y = df[target]
            
            # Verify shapes
            if len(X) != len(y):
                st.error(f"Shape mismatch for {target}: X has {len(X)} samples, y has {len(y)}")
                continue
                
            # Feature scaling
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            
            # Initialize models
            ridge = Ridge(alpha=1.0)
            lasso = Lasso(alpha=0.01, max_iter=10000)
            rf_reg = RandomForestRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=10,
                random_state=42
            )
            
            # Cross-validation
            cv_scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
            cv_r2 = np.mean(cv_scores)
            
            # Train models
            ridge.fit(X_train, y_train)
            lasso.fit(X_train, y_train)
            rf_reg.fit(X_train, y_train)
            
            # Calculate validation metrics
            val_pred = ridge.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            # Store results
            models[target] = {
                'ridge': ridge,
                'lasso': lasso,
                'rf_reg': rf_reg,
                'features': X.columns.tolist(),
                'cv_r2': cv_r2,
                'val_r2': val_r2,
                'val_mse': val_mse,
                'scaler': scaler,
                'n_samples': len(X)
            }
            
        return models
    
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

# ----------------------------
# Visualization Functions
# ----------------------------
def plot_feature_importance(model_data, target):
    features = model_data['features']
    importance = model_data['rf_reg'].feature_importances_
    
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
    
    fig = px.box(
        df[waste_cols],
        title="Waste Distribution Across Provinces",
        points="all"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Main Application
# ----------------------------
def main():
    st.title("🇹🇭 Thailand Solid Waste Prediction System")
    st.markdown("""
    This system predicts waste generation patterns across Thailand's provinces using machine learning models.
    """)
    
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        st.stop()
    
    # Train models
    models = train_all_waste_models(df)
    if models is None:
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Analysis", "🗂️ Data Explorer"])
    
    # Prediction Tab
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
            # Prepare input
            X_input = pd.DataFrame([input_data])
            X_input_scaled = models[target]['scaler'].transform(X_input)
            
            # Get predictions
            ridge_pred = models[target]['ridge'].predict(X_input_scaled)[0]
            lasso_pred = models[target]['lasso'].predict(X_input_scaled)[0]
            rf_pred = models[target]['rf_reg'].predict(X_input_scaled)[0]
            
            # Display results
            st.success("### Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Ridge Regression Prediction", f"{ridge_pred:.2f} tons/day")
                st.metric("Lasso Regression Prediction", f"{lasso_pred:.2f} tons/day")
                st.metric("Random Forest Prediction", f"{rf_pred:.2f} tons/day")
            
            with col2:
                st.write("#### Model Performance")
                st.write(f"- Cross-Validated R²: {models[target]['cv_r2']:.3f}")
                st.write(f"- Validation R²: {models[target]['val_r2']:.3f}")
                st.write(f"- Validation MSE: {models[target]['val_mse']:.3f}")
    
    # Analysis Tab
    with tab2:
        st.header("Model Analysis")
        target = st.selectbox("Select Waste Type for Analysis", options=list(models.keys()), key='analysis')
        
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cross-Validated R²", f"{models[target]['cv_r2']:.3f}")
        col2.metric("Validation R²", f"{models[target]['val_r2']:.3f}")
        col3.metric("Validation MSE", f"{models[target]['val_mse']:.3f}")
        
        st.subheader("Feature Importance")
        plot_feature_importance(models[target], target)
    
    # Data Explorer Tab
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
        
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

if __name__ == "__main__":
    main()
