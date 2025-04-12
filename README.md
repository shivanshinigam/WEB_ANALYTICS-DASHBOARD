# 📊 Web Analytics Dashboard

This is an interactive **Streamlit** dashboard for web analytics, designed to analyze web traffic data with various machine learning models, anomaly detection, and forecasting techniques. It includes:

- Time-series forecasting models (ARIMA, SARIMA, VAR)
- Machine learning regression models (Random Forest, XGBoost)
- Anomaly detection with Isolation Forest
- Dimensionality reduction with **SVD** (Singular Value Decomposition)
- Clustering websites/pages with **KMeans**
- Data visualization for key metrics (Page Views, Bounce Rate, Exit Rate)

## 🚀 Features

- 📈 **Time-Series Forecasting**: Predicts future page views using ARIMA, SARIMA, and VAR models.
- 🌲 **Machine Learning Models**: Uses Random Forest and XGBoost to predict page views based on other web metrics.
- 🚨 **Anomaly Detection**: Uses **Isolation Forest** for detecting anomalies in web traffic metrics like page views, bounce rates, and exit rates.
- 🔍 **Filtering**: Allows filtering data by year and website for better insights.
- 🔍 **Dimensionality Reduction**: **SVD (Singular Value Decomposition)** is used to reduce the dimensionality of the dataset and visually represent relationships between websites/pages.
- 🧑‍🤝‍🧑 **Clustering**: **KMeans clustering** groups websites or pages based on traffic behavior, helping to identify similar website categories or trends.
- 📊 **Interactive Visualizations**: Provides visualizations like page views, bounce rates, exit rates, and top pages.

## 🧑‍💻 What We Do

### **1. Anomaly Detection with Isolation Forest**
   - We use **Isolation Forest** to detect anomalous behavior in web traffic data. Anomalies are identified in the **Page Views**, **Bounce Rate**, and **Exit Rate** metrics, allowing us to detect abnormal spikes or drops in traffic, which could signify unusual events (e.g., page failures or bot activity).

### **2. Predictive Models**
   - **Random Forest** and **XGBoost** are trained to predict page views based on various metrics, such as bounce rate and time on page. These models help forecast traffic patterns and provide a more accurate picture of future web traffic.

### **3. Time-Series Forecasting (ARIMA, SARIMA, VAR)**
   - **ARIMA (Auto-Regressive Integrated Moving Average)** and **SARIMA (Seasonal ARIMA)** are used for forecasting future web traffic based on historical data. 
   - **VAR (Vector Autoregression)** is applied for multivariate time series forecasting to predict multiple related metrics (e.g., Page Views, Bounce Rate, Exit Rate) together.

### **4. Dimensionality Reduction with SVD**
   - We use **SVD** (Singular Value Decomposition) to reduce the dimensionality of the data, enabling us to visualize how different websites/pages relate to each other in a lower-dimensional space. This helps in identifying patterns or similarities between different pages/websites in terms of traffic behavior.

### **5. Clustering with KMeans**
   - **KMeans clustering** is used to group websites or pages into clusters based on similar traffic patterns. For instance, websites with similar bounce rates, exit rates, and page views will be grouped together, enabling us to analyze common behaviors across different website categories.

📷 Screenshots
<img width="1644" alt="Screenshot 2025-04-12 at 11 15 28 PM" src="https://github.com/user-attachments/assets/8eccfa82-1e57-43f2-b6f1-8f213103c425" />

<img width="1633" alt="Screenshot 2025-04-12 at 11 15 50 PM" src="https://github.com/user-attachments/assets/c6d78a1d-9701-47ba-9a62-11305fcd4a57" />

<img width="1628" alt="Screenshot 2025-04-12 at 11 16 19 PM" src="https://github.com/user-attachments/assets/505421d5-852f-468a-899c-8839d94406aa" />


