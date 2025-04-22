Web Analytics Dashboard

A powerful, interactive Web Analytics Dashboard built with Streamlit to analyze web performance metrics, detect anomalies, and forecast trends using advanced machine learning and time-series models. This tool transforms raw web analytics data into actionable insights for marketers, data analysts, and business owners.

🚀 Features

📋 Overview: Displays sample data and key metrics like total page views, average bounce rate, and time on page.
📈 Trends: Visualizes yearly and website-specific trends with interactive line and bar charts.
🚨 Anomaly Detection: Identifies outliers in page views, bounce rate, and exit rate using Isolation Forest.
📊 Exploratory Analysis: Highlights top pages, bounce rate distributions, and relationships between metrics.
📉 Predictive Analytics: Forecasts page views using Random Forest, XGBoost, ARIMA, SARIMA, VAR, and LSTM models.
📑 Project Summary: Provides a comprehensive overview of the dashboard’s purpose and methodologies.
⚙️ Interactive Filters: Customize analysis by year and website with a user-friendly sidebar.
📊 Visualizations: Powered by Plotly for dynamic, interactive charts.


🎯 Objectives

Visualize Key Metrics: Present critical web analytics data in an intuitive format.
Detect Anomalies: Flag unusual patterns for further investigation.
Enable Exploration: Support deep dives into data with interactive tools.
Predict Trends: Forecast future performance to guide strategic decisions.
Empower Users: Deliver actionable insights for technical and non-technical stakeholders.


🛠️ Tech Stack

Frontend: Streamlit
Data Processing: Pandas, NumPy
Visualization: Plotly Express
Machine Learning: Scikit-learn, XGBoost
Time-Series Forecasting: Statsmodels (ARIMA, SARIMA, VAR)
Deep Learning: Keras (LSTM)
Environment: Python 3.8+


📂 Project Structure
web-analytics-dashboard/
│
├── app.py                  # Main Streamlit application script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── sample_data/            # Sample CSV data (optional)
│   └── web_analytics.csv
└── assets/                 # Images and other assets for README
    └── dashboard_screenshot.png


🧑‍💻 Installation
Follow these steps to set up the project locally:
Prerequisites

Python 3.8 or higher
Git
Virtual environment tool (e.g., venv or virtualenv)

Steps

Clone the Repository:
git clone https://github.com/your-username/web-analytics-dashboard.git
cd web-analytics-dashboard


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Run the Application:
streamlit run app.py


Access the Dashboard:Open your browser and navigate to http://localhost:8501.



📊 Usage

Upload Data:

Use the sidebar to upload a CSV file containing web analytics data.
Expected columns: YEAR, WEBSITE, PAGE_URL, PAGE_VIEWS, BOUNCE_RATE_(%), EXIT_RATE_(%), AVERAGE_TIME_ON_PAGE_(SECONDS).


Apply Filters:

Select specific years and websites to tailor the analysis.
Reset filters to default with the "Reset Filters" button.


Explore Tabs:

Overview: View sample data and summary statistics.
Trends: Analyze temporal and website-specific trends.
Anomaly Detection: Adjust sensitivity and inspect outliers.
Exploratory Analysis: Dive into top pages and metric distributions.
Predictive Analytics: Compare model performance and view forecasts.
Project Summary: Read about the dashboard’s features and methodologies.


Interact with Visualizations:

Hover over charts to see details.
Zoom and pan on Plotly charts for deeper exploration.




📈 Sample Data
A sample CSV (web_analytics.csv) can be used for testing. The data should include:



YEAR
WEBSITE
PAGE_URL
PAGE_VIEWS
BOUNCE_RATE_(%)
EXIT_RATE_(%)
AVERAGE_TIME_ON_PAGE_(SECONDS)



2023
example.com
/home
10000
45.5
30.2
120


2023
example.com
/products
8000
50.0
35.0
90



🧠 Methodologies

Data Preprocessing: Cleans column names, handles missing values, and applies scaling (StandardScaler, MinMaxScaler).
Anomaly Detection: Uses Isolation Forest with adjustable contamination for outlier detection.
Predictive Modeling:
Random Forest/XGBoost: Ensemble methods for regression.
ARIMA/SARIMA: Time-series models for univariate forecasting.
VAR: Multivariate time-series forecasting.
LSTM: Deep learning for complex temporal patterns.


Evaluation: Models are compared using Mean Absolute Error (MAE).
Visualization: Plotly Express for interactive line, bar, scatter, and histogram charts.


🌟 Why This Dashboard?

Interactive: Streamlit-based UI for a seamless experience.
Customizable: Dynamic filters for tailored analysis.
Comprehensive: Combines descriptive, diagnostic, and predictive analytics.
Scalable: Handles large datasets with efficient processing.
Cutting-Edge: Leverages state-of-the-art machine learning and forecasting techniques.


🔮 Future Enhancements

Real-Time Data: Integrate APIs for live analytics.
Advanced Segmentation: Add filters for demographics or traffic sources.
Custom Model Tuning: Allow users to adjust model hyperparameters.
Exportable Reports: Support PDF/Excel exports.
A/B Testing: Compare page performance for experiments.


🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please follow the Code of Conduct.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙌 Acknowledgments

Streamlit: For an amazing framework to build interactive apps.
Plotly: For powerful visualization tools.
Scikit-learn, Statsmodels, XGBoost, Keras: For robust machine learning and forecasting libraries.
Pandas/NumPy: For efficient data processing.


📬 Contact
For questions or feedback, reach out via shivanshinigam4@gmail.com or open an issue on GitHub.

Built with 💻 by SHIVANSHI NIGAM

Happy analyzing! 🚀
