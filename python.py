import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense

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

    if st.sidebar.button("🔄 Reset Filters"):
        st.session_state.selected_years = years
        st.session_state.selected_websites = websites

    st.session_state.selected_years = selected_years
    st.session_state.selected_websites = selected_websites

    df = df[df['YEAR'].isin(selected_years) & df['WEBSITE'].isin(selected_websites)]

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

        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X, y)
        rf_predictions = rf_model.predict(X)
        rf_mae = np.mean(np.abs(rf_predictions - y))
        st.metric("Random Forest MAE", round(rf_mae, 2))

        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(X, y)
        xgb_predictions = xgb_model.predict(X)
        xgb_mae = np.mean(np.abs(xgb_predictions - y))
        st.metric("XGBoost MAE", round(xgb_mae, 2))

        arima_model = ARIMA(y, order=(5,1,0))
        arima_model_fit = arima_model.fit()
        arima_predictions = arima_model_fit.predict(start=0, end=len(y)-1, typ='levels')
        arima_mae = np.mean(np.abs(arima_predictions - y))
        st.metric("ARIMA MAE", round(arima_mae, 2))

        sarima_model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
        sarima_model_fit = sarima_model.fit(disp=False)
        sarima_predictions = sarima_model_fit.predict(start=0, end=len(y)-1)
        sarima_mae = np.mean(np.abs(sarima_predictions - y))
        st.metric("SARIMA MAE", round(sarima_mae, 2))

        st.subheader("📉 VAR Model")
        df_var = df[['PAGE_VIEWS', 'BOUNCE_RATE_(%)', 'EXIT_RATE_(%)']]
        def test_stationarity(df):
            return {col: adfuller(df[col].dropna())[1] for col in df.columns}

        df_var_diff = df_var.diff().dropna()
        model = VAR(df_var_diff)
        lag_order = model.select_order(maxlags=10).aic
        var_model = model.fit(lag_order)
        forecast_steps = 5
        forecast = var_model.forecast(df_var_diff.values[-lag_order:], steps=forecast_steps)
        forecast_df = pd.DataFrame(forecast, index=range(df_var.index[-1]+1, df_var.index[-1]+forecast_steps+1),
                                   columns=df_var.columns)
        fig7 = px.line(forecast_df, title="VAR Forecast")
        st.plotly_chart(fig7)
        actual_data = df_var[-forecast_steps:]
        mae_var = np.mean(np.abs(forecast_df.values - actual_data.values))

        st.subheader("LSTM Model")
        lstm_df = df[['PAGE_VIEWS']].fillna(method='ffill')
        scaler_lstm = MinMaxScaler()
        scaled_data = scaler_lstm.fit_transform(lstm_df)

        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)

        seq_length = 10
        X_lstm, y_lstm = create_sequences(scaled_data, seq_length)
        split_idx = int(len(X_lstm) * 0.8)
        X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

        model_lstm = Sequential([
            LSTM(50, return_sequences=False, input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
        predictions = model_lstm.predict(X_test)
        predictions = scaler_lstm.inverse_transform(predictions)
        actual = scaler_lstm.inverse_transform(y_test.reshape(-1, 1))
        lstm_mae = np.mean(np.abs(predictions.flatten() - actual.flatten()))

        lstm_results = pd.DataFrame({"Actual": actual.flatten(), "Predicted": predictions.flatten()})
        fig_lstm = px.line(lstm_results, title="LSTM Prediction vs Actual")
        st.plotly_chart(fig_lstm)
        st.metric("LSTM MAE", round(lstm_mae, 2))

        model_comparison = pd.DataFrame({
            "Model": ["Random Forest", "XGBoost", "ARIMA", "SARIMA", "VAR", "LSTM"],
            "MAE": [rf_mae, xgb_mae, arima_mae, sarima_mae, mae_var, lstm_mae]
        })
        st.write(model_comparison)

    with tab6:
        st.subheader("📑 Project Summary")
        st.write("""### 🚀 Overview
    Welcome to the **Web Analytics Dashboard**, a powerful and interactive tool designed to empower stakeholders with actionable insights into website performance. Built using **Streamlit**, this dashboard leverages advanced data analytics, machine learning, and time-series forecasting to transform raw web analytics data into meaningful visualizations and predictions. Whether you're a marketer, data analyst, or business owner, this tool provides a comprehensive view of key performance metrics, uncovers anomalies, and forecasts future trends to support data-driven decision-making.

    ---

    ### 🎯 Objectives
    The Web Analytics Dashboard aims to:
    1. **Visualize Key Metrics**: Provide intuitive visualizations of critical web analytics metrics such as page views, bounce rate, exit rate, and average time on page.
    2. **Detect Anomalies**: Identify unusual patterns in website performance to flag potential issues or opportunities.
    3. **Enable Exploratory Analysis**: Allow users to explore data through interactive filters and detailed charts to uncover trends and relationships.
    4. **Predict Future Trends**: Use advanced machine learning and time-series models to forecast page views and other metrics.
    5. **Empower Stakeholders**: Deliver a user-friendly interface with actionable insights for both technical and non-technical users.

    ---

    ### 🛠️ Key Features
    The dashboard is organized into six intuitive tabs, each addressing a specific aspect of web analytics:

    1. **📋 Overview**
       - Displays a sample of the uploaded dataset and summary statistics.
       - Key metrics include total page views, average bounce rate, and average time on page, presented in a clean, metric-card format.

    2. **📈 Trends**
       - Visualizes temporal trends (e.g., yearly page views) and website-specific performance (e.g., total page views by website).
       - Uses Plotly for interactive line and bar charts.

    3. **🚨 Anomaly Detection**
       - Employs the **Isolation Forest** algorithm to detect anomalies in page views, bounce rate, and exit rate.
       - Includes a customizable sensitivity slider for fine-tuning anomaly detection.
       - Visualizes anomalies in a scatter plot for easy interpretation.

    4. **📊 Exploratory Analysis**
       - Highlights top-performing pages by page views.
       - Provides distributions (e.g., bounce rate histogram) and relationships (e.g., exit rate vs. bounce rate scatter plot).
       - Enables deep dives into page-level performance.

    5. **📉 Predictive Analytics**
       - Implements multiple predictive models to forecast page views:
         - **Random Forest** and **XGBoost** for regression-based predictions.
         - **ARIMA** and **SARIMA** for univariate time-series forecasting.
         - **VAR** for multivariate time-series forecasting.
         - **LSTM** for deep learning-based time-series predictions.
       - Compares model performance using Mean Absolute Error (MAE) in a tabular format.
       - Visualizes forecasts and actual vs. predicted values.

    6. **📑 Project Summary**
       - You’re here! A comprehensive summary of the dashboard’s purpose, features, and methodologies.

    ---

    ### 🧠 Methodologies
    The dashboard employs a robust suite of data science techniques to deliver reliable insights:

    - **Data Preprocessing**:
      - Handles missing values and standardizes column names for consistency.
      - Applies **StandardScaler** and **MinMaxScaler** for feature scaling in anomaly detection and predictive modeling.

    - **Visualization**:
      - Uses **Plotly Express** for interactive and visually appealing charts, including line plots, bar charts, histograms, and scatter plots.

    - **Anomaly Detection**:
      - Utilizes **Isolation Forest**, an unsupervised machine learning algorithm, to identify outliers in key metrics.
      - Allows users to adjust the contamination parameter for sensitivity control.

    - **Predictive Modeling**:
      - **Random Forest** and **XGBoost**: Ensemble methods for robust regression on page views using bounce rate, exit rate, and time on page as features.
      - **ARIMA/SARIMA**: Time-series models for capturing trends and seasonality in page views.
      - **VAR**: Multivariate time-series model to account for interdependencies between page views, bounce rate, and exit rate.
      - **LSTM**: A deep learning approach for capturing complex temporal patterns in page views.
      - Model performance is evaluated using **Mean Absolute Error (MAE)** to ensure accuracy.

    - **Exploratory Analysis**:
      - Leverages **Pandas** for data aggregation and **Plotly** for visualizing distributions and relationships.
      - Focuses on actionable insights, such as identifying top pages and understanding bounce rate patterns.

    ---

    ### 📊 Insights & Applications
    The Web Analytics Dashboard delivers insights that can drive strategic decisions:
    - **Performance Monitoring**: Track page views and engagement metrics over time to assess website health.
    - **Anomaly Identification**: Quickly detect underperforming pages or unusual spikes in bounce/exit rates for further investigation.
    - **Trend Analysis**: Understand seasonal patterns and website-specific performance to optimize content and marketing strategies.
    - **Predictive Planning**: Use forecasts to anticipate traffic trends and allocate resources effectively.
    - **User Engagement**: Identify high-performing pages and replicate their success across the website.

    Applications include:
    - **Marketing Campaigns**: Evaluate the impact of campaigns on page views and engagement.
    - **Content Optimization**: Prioritize content updates for pages with high bounce or exit rates.
    - **SEO Strategy**: Use anomaly detection to flag pages needing SEO improvements.
    - **Resource Allocation**: Plan server capacity and marketing budgets based on predicted traffic.

    ---

    ### 🌟 Why This Dashboard?
    - **Interactive & User-Friendly**: Built with Streamlit for a seamless, browser-based experience.
    - **Customizable**: Dynamic filters for years and websites allow tailored analysis.
    - **Comprehensive**: Combines descriptive, diagnostic, and predictive analytics in one tool.
    - **Scalable**: Handles large datasets and supports CSV uploads for flexibility.
    - **Cutting-Edge**: Integrates state-of-the-art machine learning and time-series models.

    ---

    ### 🔮 Future Enhancements
    To make the dashboard even more powerful, future updates could include:
    - **Real-Time Data Integration**: Connect to APIs for live web analytics data.
    - **Advanced Segmentation**: Add filters for user demographics, traffic sources, or device types.
    - **Custom Model Tuning**: Allow users to adjust hyperparameters for predictive models.
    - **Exportable Reports**: Enable PDF or Excel exports of visualizations and insights.
    - **A/B Testing Support**: Integrate tools to compare page performance for A/B tests.

    ---

    ### 🙌 Get Started
    Upload your web analytics data (CSV format) using the sidebar, apply filters, and explore the tabs to unlock insights. Whether you're optimizing a single website or managing a portfolio, this dashboard is your go-to tool for understanding and enhancing web performance.

    **Built with 💻 by SHIVANSHI NIGAM**  
    Powered by **Streamlit**, **Pandas**, **Plotly**, **Scikit-learn**, **Statsmodels**, **XGBoost**, and **Keras**.""")
else:
    st.warning("⚠️ Please upload a CSV file to begin.")
