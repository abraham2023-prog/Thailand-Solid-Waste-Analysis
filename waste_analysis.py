# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Waste Prediction System",
    page_icon="🗑️",
    layout="wide"
)

# ----------------------------
# Data Loading & Preparation
# ----------------------------
@st.cache_data
def load_and_prepare_data():
    try:
        # Build the absolute path to the CSV file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'SW_Thailand_2021_Labeled.csv')
        df = pd.read_csv(csv_path)
        
        # Data Quality Report
        with st.expander("🔍 Data Quality Report", expanded=False):
            st.write("### Missing Values Before Imputation")
            missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])
            st.dataframe(missing_df.style.background_gradient(cmap='Reds'))
            
            # Check for constant columns
            constant_cols = [col for col in df.columns if df[col].nunique() == 1]
            if constant_cols:
                st.warning(f"Constant columns detected and removed: {constant_cols}")
                df = df.drop(columns=constant_cols)
        
        # Feature Engineering
        waste_targets = ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']
        base_features = ['Pop', 'GPP_per_Capita', 'GPP_Industrial(%)', 
                       'Visitors(ppl)', 'GPP_Agriculture(%)', 
                       'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)']
        
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
            st.warning(f"Dropping highly correlated features: {to_drop}")
            base_features = [f for f in base_features if f not in to_drop]
        
        # Final feature selection
        features = [f for f in base_features if f in df.columns]
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])
        df = df.dropna(subset=waste_targets, how='all')
        
        return df[features + waste_targets]
    
    except Exception as e:
        st.error(f"❌ Data loading failed: {str(e)}")
        return None

# ----------------------------
# Model Training
# ----------------------------
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
                
            # Prepare features
            other_targets = [wt for wt in waste_targets if wt != target]
            X = df.drop(columns=other_targets)
            y = df[target]
            
            # Skip if not enough data
            if len(y.dropna()) < 20:
                st.warning(f"Not enough data for {target}")
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
                max_features='sqrt',
                random_state=42
            )
            
            # Cross-validation
            cv_scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
            cv_r2 = np.mean(cv_scores)
            
            # Train models
            ridge.fit(X_train, y_train)
            lasso.fit(X_train, y_train)
            rf_reg.fit(X_train, y_train)
            
            # Classification setup
            threshold = y.quantile(0.75)
            y_clf = (y > threshold).astype(int)
            rf_clf = None
            
            if y_clf.nunique() >= 2 and min(y_clf.value_counts()) >= 5:
                rf_clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=3,
                    min_samples_leaf=10,
                    class_weight='balanced',
                    random_state=42
                )
                rf_clf.fit(X_train, y_clf)
            
            # Store models and metrics
            models[target] = {
                'ridge': ridge,
                'lasso': lasso,
                'rf_reg': rf_reg,
                'rf_clf': rf_clf,
                'features': X.columns.tolist(),
                'threshold': threshold,
                'cv_r2': cv_r2,
                'scaler': scaler,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }
            
        return models
    
    except Exception as e:
        st.error(f"❌ Model training failed: {str(e)}")
        return None

# ----------------------------
# Visualization Functions
# ----------------------------
def plot_feature_importance(models, target):
    fig = px.bar(
        x=models[target]['features'],
        y=models[target]['rf_reg'].feature_importances_,
        title=f"Feature Importance for {target}",
        labels={'x': 'Features', 'y': 'Importance'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def plot_waste_distribution(df):
    waste_cols = [col for col in ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste'] 
                 if col in df.columns]
    
    fig = px.box(
        df[waste_cols],
        title="Waste Distribution Across Provinces"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("🇹🇭 Thailand Waste Prediction System")
    st.markdown("""
    This system predicts waste generation patterns across Thailand's provinces using machine learning models.
    """)
    
    # Load data and models
    df = load_and_prepare_data()
    models = train_all_waste_models()
    
    if df is None or models is None:
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Predict Waste",
        "📈 Model Analysis",
        "🗂️ Data Explorer",
        "ℹ️ About"
    ])
    
    # --- Prediction Tab ---
    with tab1:
        st.header("Make Waste Predictions")
        target = st.selectbox("Select Waste Type", options=list(models.keys()))
        
        cols = st.columns(3)
        input_data = {}
        
        # Dynamic input sliders
        for i, feature in enumerate(models[target]['features']):
            if feature in df.columns:
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
            X_input_scaled = models[target]['scaler'].transform(X_input)
            
            # Get predictions
            ridge_pred = models[target]['ridge'].predict(X_input_scaled)[0]
            lasso_pred = models[target]['lasso'].predict(X_input_scaled)[0]
            rf_pred = models[target]['rf_reg'].predict(X_input_scaled)[0]
            
            # Display results
            st.success("### Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Ridge Regression", f"{ridge_pred:.2f} tons/day")
                st.metric("Lasso Regression", f"{lasso_pred:.2f} tons/day")
                st.metric("Random Forest", f"{rf_pred:.2f} tons/day")
                
            with col2:
                if models[target]['rf_clf'] is not None:
                    clf_pred = models[target]['rf_clf'].predict(X_input_scaled)[0]
                    clf_proba = models[target]['rf_clf'].predict_proba(X_input_scaled)[0]
                    
                    st.metric("Waste Level", 
                             "High" if clf_pred == 1 else "Low",
                             f"Confidence: {max(clf_proba)*100:.1f}%")
                    
                    # Show probability distribution
                    prob_df = pd.DataFrame({
                        'Probability': clf_proba,
                        'Class': ['Low Waste', 'High Waste']
                    })
                    fig = px.bar(prob_df, x='Class', y='Probability', 
                                title="Classification Confidence")
                    st.plotly_chart(fig, use_container_width=True)
    
    # --- Analysis Tab ---
    with tab2:
        st.header("Model Performance Analysis")
        target = st.selectbox("Select Waste Type", options=list(models.keys()), key='analysis')
        
        st.subheader("Regression Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Cross-Validated R²", f"{models[target]['cv_r2']:.3f}")
        col2.metric("Classification Threshold", f"{models[target]['threshold']:.2f} tons/day")
        
        st.subheader("Feature Importance")
        plot_feature_importance(models, target)
        
        if models[target]['rf_clf'] is not None:
            st.subheader("Classification Report")
            y_pred = models[target]['rf_clf'].predict(models[target]['X_val'])
            st.text(classification_report(
                models[target]['y_val'] > models[target]['threshold'],
                y_pred
            ))
    
    # --- Data Explorer Tab ---
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
    
    # --- About Tab ---
    with tab4:
        st.header("About This Project")
        st.markdown("""
        This waste prediction system was developed to help policymakers and environmental agencies:
        - Predict waste generation patterns
        - Identify key drivers of different waste types
        - Optimize waste management strategies
        
        **Data Source:** Thailand Provincial Waste Data (2021)
        **Models Used:** Ridge/Lasso Regression, Random Forest
        """)

if __name__ == "__main__":
    main()
