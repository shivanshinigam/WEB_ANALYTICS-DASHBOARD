import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from prophet import Prophet
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Streamlit Page Config
st.set_page_config(page_title="Web Analytics Dashboard", layout="wide")
st.title("📊 Web Analytics Dashboard")
st.sidebar.header("⚙️ Upload & Settings")

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your web analytics data (CSV)", type="csv")

# Default states for filter options in session_state
if 'selected_years' not in st.session_state:
    st.session_state.selected_years = []
if 'selected_websites' not in st.session_state:
    st.session_state.selected_websites = []

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean Column Names
    df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

    # Sidebar Filters
    st.sidebar.subheader("🔍 Filters")
    years = sorted(df['YEAR'].dropna().unique())
    selected_years = st.sidebar.multiselect("Select Year(s)", years, default=st.session_state.selected_years or years)

    websites = sorted(df['WEBSITE'].dropna().unique())
    selected_websites = st.sidebar.multiselect("Select Website(s)", websites, default=st.session_state.selected_websites or websites)

    # Reset button
    if st.sidebar.button("🔄 Reset Filters"):
        st.session_state.selected_years = years
        st.session_state.selected_websites = websites

    # Update session state on filter change
    st.session_state.selected_years = selected_years
    st.session_state.selected_websites = selected_websites

    # Filtered Data
    df = df[df['YEAR'].isin(selected_years) & df['WEBSITE'].isin(selected_websites)]

    # 👇 Tabs must be defined BEFORE use
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Overview", 
        "📈 Trends", 
        "🚨 Anomaly Detection", 
        "📊 Exploratory Analysis",
        "📉 Predictive Analytics",
        "📑 Project Summary"
    ])

    with tab1:
        st.subheader("📋 Sample Data")
        st.dataframe(df.head())

        st.subheader("📌 Summary Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Page Views", int(df['PAGE_VIEWS'].sum()))
        col2.metric("Avg Bounce Rate (%)", round(df['BOUNCE_RATE_(%)'].mean(), 2))
        col3.metric("Avg Time on Page (s)", round(df['AVERAGE_TIME_ON_PAGE_(SECONDS)'].mean(), 2))

    with tab2:
        st.subheader("📈 Page Views by Year")
        df_yearly = df.groupby('YEAR', as_index=False)['PAGE_VIEWS'].sum()
        fig1 = px.line(df_yearly, x='YEAR', y='PAGE_VIEWS', markers=True, title="Yearly Page Views")
        st.plotly_chart(fig1)

        st.subheader("📊 Page Views by Website")
        df_site = df.groupby('WEBSITE', as_index=False)['PAGE_VIEWS'].sum()
        fig2 = px.bar(df_site, x='WEBSITE', y='PAGE_VIEWS', title="Total Page Views by Website")
        st.plotly_chart(fig2)

    with tab3:
        st.subheader("🚨 Anomaly Detection on Key Metrics")

        features = ['PAGE_VIEWS', 'BOUNCE_RATE_(%)', 'EXIT_RATE_(%)']
        df_filtered = df[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_filtered)

        contamination = st.sidebar.slider("🔍 Anomaly Sensitivity", 0.01, 0.2, 0.05)
        iso = IsolationForest(contamination=contamination, random_state=42)
        df['Anomaly'] = iso.fit_predict(X_scaled)

        anomalies = df[df['Anomaly'] == -1]
        st.metric("Anomalous Pages Detected", len(anomalies))
        st.dataframe(anomalies[['PAGE_URL', 'PAGE_VIEWS', 'BOUNCE_RATE_(%)', 'EXIT_RATE_(%)']])

        st.subheader("📌 Anomaly Scatter Plot")
        fig3 = px.scatter(df, x='PAGE_VIEWS', y='BOUNCE_RATE_(%)', 
                          color=df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'}),
                          title="Anomaly Detection: Page Views vs Bounce Rate")
        st.plotly_chart(fig3)

    with tab4:
        st.subheader("📊 Top Pages by Page Views")
        top_pages = df.sort_values(by='PAGE_VIEWS', ascending=False).head(10)
        fig4 = px.bar(top_pages, x='PAGE_URL', y='PAGE_VIEWS', title="Top 10 Pages", text='PAGE_VIEWS')
        fig4.update_traces(textposition='outside')
        st.plotly_chart(fig4)

        st.subheader("📈 Bounce Rate Distribution")
        fig5 = px.histogram(df, x='BOUNCE_RATE_(%)', nbins=30, title="Bounce Rate Histogram")
        st.plotly_chart(fig5)

        st.subheader("📉 Exit Rate vs Bounce Rate")
        fig6 = px.scatter(df, x='EXIT_RATE_(%)', y='BOUNCE_RATE_(%)', 
                          hover_data=['PAGE_URL'], title="Exit Rate vs Bounce Rate")
        st.plotly_chart(fig6)

    with tab5:
        st.subheader("📉 Predictive Analytics")

        features = ['BOUNCE_RATE_(%)', 'EXIT_RATE_(%)', 'AVERAGE_TIME_ON_PAGE_(SECONDS)']
        df_filtered = df[features].fillna(0)
        X = df_filtered
        y = df['PAGE_VIEWS']

        # Random Forest Model
        st.subheader("🌲 Random Forest Regression")
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X, y)
        rf_predictions = rf_model.predict(X)
        rf_mae = np.mean(np.abs(rf_predictions - y))
        st.metric("Random Forest MAE", round(rf_mae, 2))

        # XGBoost Model
        st.subheader("🚀 XGBoost Regression")
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(X, y)
        xgb_predictions = xgb_model.predict(X)
        xgb_mae = np.mean(np.abs(xgb_predictions - y))
        st.metric("XGBoost MAE", round(xgb_mae, 2))

        # ARIMA Model
        st.subheader("📉 ARIMA Model")
        arima_model = ARIMA(y, order=(5,1,0))  # Adjust the ARIMA order based on your data
        arima_model_fit = arima_model.fit()
        arima_predictions = arima_model_fit.predict(start=0, end=len(y)-1, typ='levels')
        arima_mae = np.mean(np.abs(arima_predictions - y))
        st.metric("ARIMA MAE", round(arima_mae, 2))

        # SARIMA Model
        st.subheader("📈 SARIMA Model")
        sarima_model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))  # Adjust as needed
        sarima_model_fit = sarima_model.fit(disp=False)
        sarima_predictions = sarima_model_fit.predict(start=0, end=len(y)-1)
        sarima_mae = np.mean(np.abs(sarima_predictions - y))
        st.metric("SARIMA MAE", round(sarima_mae, 2))

        

        # VAR Model (New Addition)
        st.subheader("📉 VAR Model (Vector Autoregression)")

    # Prepare the data for VAR
        df_var = df[['PAGE_VIEWS', 'BOUNCE_RATE_(%)', 'EXIT_RATE_(%)']]
        
        # Check for stationarity (ADF test)
        def test_stationarity(df):
            p_values = {}
            for column in df.columns:
                result = adfuller(df[column].dropna())
                p_values[column] = result[1]
            return p_values

        stationarity_results = test_stationarity(df_var)
        st.write("Stationarity test results (p-values):", stationarity_results)

        # If p-values are greater than 0.05, the data is not stationary, so we differencing it
        df_var_diff = df_var.diff().dropna()

        # Check stationarity of differenced data
        stationarity_results_diff = test_stationarity(df_var_diff)
        st.write("Stationarity test results after differencing (p-values):", stationarity_results_diff)

        # Fit the VAR model
        model = VAR(df_var_diff)
        lag_order = model.select_order(maxlags=10).aic  # Use AIC to select the best lag length
        var_model = model.fit(lag_order)
        st.write(f"Optimal lag length: {lag_order}")

        # Forecasting with the VAR model (next 5 years)
        forecast_steps = 5
        forecast = var_model.forecast(df_var_diff.values[-lag_order:], steps=forecast_steps)

        # Convert forecast to DataFrame for visualization
        forecast_df = pd.DataFrame(forecast, index=range(df_var.index[-1] + 1, df_var.index[-1] + forecast_steps + 1),
                                columns=df_var.columns)
        st.write(f"Forecast for next {forecast_steps} years:", forecast_df)

        # Visualize the forecast for each variable
        fig7 = px.line(forecast_df, title="VAR Forecast")
        st.plotly_chart(fig7)

        # Model Comparison (Calculating MAE for VAR)
        actual_data = df_var[-forecast_steps:]  # Use actual data for the forecast period
        mae_var = np.mean(np.abs(forecast_df.values - actual_data.values))  # Calculate MAE

        model_comparison = pd.DataFrame({
            "Model": ["Random Forest", "XGBoost", "ARIMA", "SARIMA", "VAR"],
            "MAE": [rf_mae, xgb_mae, arima_mae, sarima_mae, mae_var]  # Added VAR MAE
        })
        st.write(model_comparison)

    with tab6:
        st.subheader("📑 Project Summary")

        st.write("""
        ## Web Analytics Dashboard Project Summary

        This project provides an interactive web analytics dashboard to explore, analyze, and model web traffic data. The application utilizes multiple **machine learning models**, **visualizations**, and **predictive analytics** to offer deep insights into the traffic behavior of websites or pages.

        ### 1. Data Analysis and Preprocessing
        - We start by uploading and cleaning the dataset, ensuring that the column names are standardized and missing values are handled. The data is then filtered based on user-selected years and websites.
        
        ### 2. Key Libraries Used:
        - **Streamlit**: The core framework used for building the interactive web application.
        - **Plotly**: Provides interactive plots to visualize the data and the results.
        - **scikit-learn**: Used for machine learning models, such as:
            - **Isolation Forest**: For anomaly detection in web traffic data.
            - **Random Forest** and **XGBoost**: For predictive regression models to estimate page views based on other web metrics.
            - **KMeans**: For clustering websites or pages into groups based on their behavior.
            - **SVD (Singular Value Decomposition)**: For dimensionality reduction to visualize the relationships between websites/pages in reduced dimensions.
        - **statsmodels**: Used for **ARIMA**, **SARIMA**, and **Holt-Winters Exponential Smoothing** time-series models.
        - **Pandas** and **NumPy**: Essential for data manipulation and performing numerical operations.

        ### 3. Features:
        - **Data Upload**: Users can upload their own web analytics CSV file for analysis.
        - **Filter Options**: Allows users to filter data by year and website. Filters can be reset with a single click.
        - **Interactive Visualizations**: The dashboard includes several visualizations:
            - **Page Views by Year**: Tracks web traffic over time.
            - **Page Views by Website**: Compares the traffic across multiple websites.
            - **Anomaly Detection**: Identifies anomalies in web traffic metrics like page views, bounce rates, etc.
            - **Top Pages**: Identifies the most visited pages based on page views.
        - **Predictive Models**: We train **Random Forest**, **XGBoost**, **ARIMA**, **SARIMA**, and **VAR** models to predict future **Page Views** based on available metrics.
        - **Dimensionality Reduction & Clustering**: Using **SVD** for visualizing relationships and **KMeans** for clustering similar websites/pages based on their traffic patterns.

        ### 4. Outcome:
        - The dashboard empowers users to gain valuable insights from their web traffic data. With advanced features like anomaly detection, predictive modeling, and clustering, users can make data-driven decisions to improve their website's performance.

        Thank you for exploring the **Web Analytics Dashboard**! 🎉
        """)
else:
    st.warning("⚠️ Please upload a CSV file to begin.")
