# 📊 Web Analytics Dashboard

An interactive and powerful **Web Analytics Dashboard** built with **Streamlit**, designed to analyze website performance, detect anomalies, and forecast trends using advanced **machine learning** and **time-series models**. Transform raw analytics data into actionable insights—perfect for marketers, data analysts, and business owners.

---

## 🚀 Features

- **📋 Overview**: View sample data and key metrics like total page views, bounce rate, and average time on page.
- **📈 Trends**: Explore yearly and website-specific trends via interactive line and bar charts.
- **🚨 Anomaly Detection**: Spot outliers in page views, bounce rates, and exit rates using **Isolation Forest**.
- **📊 Exploratory Analysis**: Highlight top-performing pages, distributions, and relationships between key metrics.
- **📉 Predictive Analytics**: Forecast future page views with models like **Random Forest**, **XGBoost**, **ARIMA**, **SARIMA**, **VAR**, and **LSTM**.
- **💑 Project Summary**: Summarizes dashboard goals and applied methodologies.
- **⚙️ Interactive Filters**: Filter data by year and website with an intuitive sidebar.
- **📈 Visualizations**: Dynamic, high-quality charts powered by **Plotly**.

---

## 🌟 Live Demo

**📍 Check the deployed app here:** [Web Analytics Dashboard](https://webanalytics-dashboard-5kgmg2kop43oozv2efuubz.streamlit.app/)

---

## 🌟 Objectives

- **Visualize Key Metrics**: Display web analytics data in an intuitive format.
- **Detect Anomalies**: Identify unusual behavior for further investigation.
- **Enable Deep Exploration**: Interactive tools for slicing and dicing data.
- **Predict Trends**: Anticipate future performance for strategic planning.
- **Empower Users**: Provide actionable insights for both technical and non-technical users.

---

## 🛠️ Tech Stack

| Component      | Technology |
|----------------|------------|
| Frontend       | Streamlit  |
| Data Processing| Pandas, NumPy |
| Visualization  | Plotly Express |
| Machine Learning | Scikit-learn, XGBoost |
| Time-Series Forecasting | Statsmodels (ARIMA, SARIMA, VAR) |
| Deep Learning  | Keras (LSTM) |
| Environment    | Python 3.8+ |

---

## 📊 Usage

### Upload Your Data
- Use the sidebar to upload a **CSV** containing web analytics data.
- Expected columns:
  ```
  YEAR, WEBSITE, PAGE_URL, PAGE_VIEWS, BOUNCE_RATE_(%), EXIT_RATE_(%), AVERAGE_TIME_ON_PAGE_(SECONDS)
  ```

### Apply Filters
- Select specific **years** and **websites** using the sidebar.
- Reset filters easily with the "Reset Filters" button.

### Explore the Tabs
- **Overview**: Sample data and summary statistics.
- **Trends**: Analyze yearly and site-specific patterns.
- **Anomaly Detection**: Adjust sensitivity and inspect detected outliers.
- **Exploratory Analysis**: Top pages, bounce rates, and metrics relationships.
- **Predictive Analytics**: Model comparisons and page view forecasting.
- **Project Summary**: Review dashboard design and methodology.

### Interact with Charts
- **Hover** to see details.
- **Zoom** and **pan** with Plotly's dynamic features.

---

## 📊 Sample Data Example

| YEAR | WEBSITE      | PAGE_URL   | PAGE_VIEWS | BOUNCE_RATE_(%) | EXIT_RATE_(%) | AVERAGE_TIME_ON_PAGE_(SECONDS) |
|------|--------------|------------|------------|-----------------|--------------|-------------------------------|
| 2023 | example.com   | /home      | 10,000     | 45.5            | 30.2         | 120                           |
| 2023 | example.com   | /products  | 8,000      | 50.0            | 35.0         | 90                            |

---

## 🧐 Methodologies

- **Data Preprocessing**: Clean column names, handle missing values, apply scaling (**StandardScaler**, **MinMaxScaler**).
- **Anomaly Detection**: Isolation Forest with tunable contamination factor.
- **Predictive Modeling**:
  - **Random Forest**, **XGBoost**: Ensemble methods for regression.
  - **ARIMA**, **SARIMA**: Univariate time-series forecasting.
  - **VAR**: Multivariate forecasting.
  - **LSTM**: Deep learning for complex sequential patterns.
- **Evaluation**: Models compared using **Mean Absolute Error (MAE)**.
- **Visualization**: Interactive plots created with **Plotly Express**.

---

## 🔮 Why Use This Dashboard?

- **Interactive**: User-friendly Streamlit interface.
- **Customizable**: Dynamic filters for personalized analysis.
- **Comprehensive**: Covers descriptive, diagnostic, and predictive analytics.
- **Scalable**: Handles large datasets efficiently.
- **State-of-the-Art**: Utilizes modern machine learning and forecasting techniques.

---

## 🔮 Future Enhancements

- Real-time data integration via APIs.
- Advanced segmentation (e.g., demographic filters, traffic source breakdown).
- Custom model tuning through user inputs.
- Exportable PDF/Excel reports.
- A/B Testing support for page performance experiments.

---

## 🙌 Acknowledgments

- **Streamlit**: For enabling rapid development of interactive apps.
- **Plotly**: For beautiful and dynamic visualizations.
- **Scikit-learn, Statsmodels, XGBoost, Keras**: For powerful ML and forecasting libraries.
- **Pandas, NumPy**: For efficient data wrangling and processing.


