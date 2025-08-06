# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import os

# Build the absolute path to the CSV file based on the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'SW_Thailand_2021_Labeled.csv')

# Cache data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(csv_path)

        # Verify required columns exist
        required_cols = [
            'Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste',
            'Pop_Density', 'GPP_per_Capita', 'GPP_Industrial(%)', 'Visitors(ppl)',
            'GPP_Agriculture(%)', 'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Missing required columns: {missing}")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Cache model training
@st.cache_resource
def train_models():
    try:
        df = load_data()
        if df is None:
            return None
            
        # Prepare features
        X = df[[
            'Pop_Density', 'GPP_per_Capita', 'GPP_Industrial(%)', 'Visitors(ppl)',
            'GPP_Agriculture(%)', 'GPP_Services(%)', 'Age_0_5', 'MSW_GenRate(ton/d)'
        ]]
        
        # Train models for each waste type
        models = {}
        for waste_type in ['Food_Waste', 'Gen_Waste', 'Recycl_Waste', 'Hazard_Waste']:
            y = df[waste_type]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Linear Regression model
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            
            # Statsmodels OLS for detailed statistics
            X_train_sm = sm.add_constant(X_train)
            sm_model = sm.OLS(y_train, X_train_sm).fit()
            
            # Evaluate model
            y_pred = lr.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            models[waste_type] = {
                'model': lr,
                'sm_model': sm_model,
                'r2': r2,
                'mse': mse,
                'features': X.columns.tolist()
            }
            
        return models

    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

# Streamlit UI
def main():
    st.set_page_config(layout="wide")
    st.title("🇹🇭 Thailand Multi-Waste Prediction System")
    st.write("Predicts waste generation across multiple categories based on economic and demographic factors")
    
    # Load models
    models = train_models()
    if models is None:
        st.stop()
    
    df = load_data()
    if df is None:
        st.stop()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Waste Prediction", "Model Analysis", "Data Exploration"])
    
    with tab1:
        st.header("Waste Generation Prediction")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pop = st.slider(
                    "Population",
                    min_value=float(df['Pop_Density'].min()),
                    max_value=float(df['Pop_Density'].max()),
                    value=float(df['Pop_Density'].median()),
                    step=1000.0
                )
                
                gpp_per_capita = st.slider(
                    "GPP per Capita",
                    min_value=float(df['GPP_per_Capita'].min()),
                    max_value=float(df['GPP_per_Capita'].max()),
                    value=float(df['GPP_per_Capita'].median()),
                    step=1000.0
                )
                
                gpp_industrial = st.slider(
                    "Industrial GDP Share (%)",
                    min_value=float(df['GPP_Industrial(%)'].min()),
                    max_value=float(df['GPP_Industrial(%)'].max()),
                    value=float(df['GPP_Industrial(%)'].median()),
                    step=0.1
                )
                
            with col2:
                visitors = st.slider(
                    "Visitors (people)",
                    min_value=float(df['Visitors(ppl)'].min()),
                    max_value=float(df['Visitors(ppl)'].max()),
                    value=float(df['Visitors(ppl)'].median()),
                    step=1000.0
                )
                
                gpp_agri = st.slider(
                    "Agriculture GDP Share (%)",
                    min_value=float(df['GPP_Agriculture(%)'].min()),
                    max_value=float(df['GPP_Agriculture(%)'].max()),
                    value=float(df['GPP_Agriculture(%)'].median()),
                    step=0.1
                )
                
                gpp_services = st.slider(
                    "Services GDP Share (%)",
                    min_value=float(df['GPP_Services(%)'].min()),
                    max_value=float(df['GPP_Services(%)'].max()),
                    value=float(df['GPP_Services(%)'].median()),
                    step=0.1
                )
                
            with col3:
                age_0_5 = st.slider(
                    "Population Age 0-5 (%)",
                    min_value=float(df['Age_0_5'].min()),
                    max_value=float(df['Age_0_5'].max()),
                    value=float(df['Age_0_5'].median()),
                    step=0.1
                )
                
                msw_gen_rate = st.slider(
                    "MSW Generation Rate (ton/day)",
                    min_value=float(df['MSW_GenRate(ton/d)'].min()),
                    max_value=float(df['MSW_GenRate(ton/d)'].max()),
                    value=float(df['MSW_GenRate(ton/d)'].median()),
                    step=0.1
                )
            
            submitted = st.form_submit_button("Predict Waste Generation")
            
            if submitted:
                try:
                    # Prepare input data
                    input_data = pd.DataFrame([[
                        pop, gpp_per_capita, gpp_industrial, visitors,
                        gpp_agri, gpp_services, age_0_5, msw_gen_rate
                    ]], columns=models['Food_Waste']['features'])
                    
                    # Make predictions
                    results = []
                    for waste_type, model_info in models.items():
                        prediction = model_info['model'].predict(input_data)[0]
                        results.append({
                            'Waste Type': waste_type,
                            'Predicted Amount': f"{prediction:.2f}",
                            'Unit': 'ton/day'
                        })
                    
                    # Display results
                    st.success("### Prediction Results")
                    results_df = pd.DataFrame(results)
                    st.table(results_df)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    
    with tab2:
        st.header("Model Analysis")
        
        selected_model = st.selectbox(
            "Select Waste Type to Analyze",
            options=list(models.keys())
        )
        
        model_info = models[selected_model]
        
        st.subheader(f"Model Performance for {selected_model}")
        st.write(f"**R-squared:** {model_info['r2']:.3f}")
        st.write(f"**Mean Squared Error:** {model_info['mse']:.3f}")
        
        st.subheader("Model Coefficients")
        coef_df = pd.DataFrame({
            'Feature': ['Intercept'] + model_info['features'],
            'Coefficient': [model_info['sm_model'].params[0]] + list(model_info['sm_model'].params[1:])
        })
        st.dataframe(coef_df.style.format({'Coefficient': '{:.4f}'}))
        
        st.subheader("Statistical Summary")
        st.text(str(model_info['sm_model'].summary()))
    
    with tab3:
        st.header("Data Exploration")
        
        st.write("### Dataset Overview")
        st.write(f"Total records: {len(df)}")
        
        st.write("### Descriptive Statistics")
        st.dataframe(df.describe())
        
        st.write("### Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
        
        st.write("### Raw Data Preview")
        st.dataframe(df.head())

if __name__ == "__main__":
    main()
