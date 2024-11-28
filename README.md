# ClimateTempAnalysis
Climate Temperature Analysis is a web application built with Streamlit, Plotly, and Statsmodels libraries in Python. This application analyzes global temperature changes from 1961 to 2019, forecasts future temperature trends using the ARIMA model, and visualizes the data interactively to enhance understanding of climate patterns.

# Key Features
## Global Temperature Analysis: 
Visualizes average temperature changes worldwide with line and bar charts.
## Case Studies: 
Provides detailed temperature analyses for specific countries like Algeria and France.
## Top & Bottom Countries: 
Identifies the top 10 and bottom 10 countries in terms of temperature change over time.
## Temperature Forecasting: 
Uses the ARIMA model to predict future temperatures with a 95% confidence interval.
## Interactive Interface: 
Displays rolling statistics, autocorrelation, and partial autocorrelation plots.
## Data Insights: 
Presents raw and preprocessed datasets for transparency and exploration.

# Tech Stack
## Frontend: 
Streamlit for an interactive web interface.
## Backend: 
Python (libraries used: Pandas, Numpy, Matplotlib, Seaborn).
## Visualization: 
Plotly for interactive plots.
## Statistical Modeling: 
Statsmodels for ARIMA forecasting and time-series analysis.

# Parameters
## Dataset: 
The application uses temperature data from 1961 to 2019.
## Regions: 
Global analysis, with focused case studies on Algeria and France.
## Models: 
Time-series analysis using ARIMA for forecasting.

# How It Works
## Data Preprocessing:
Drops unnecessary columns and filters relevant temperature-related records.
Transforms data for visualization and analysis.
## Visualization:
Creates interactive charts to compare global and country-specific temperature changes.
Highlights decade-wise changes for easy interpretation.
## Forecasting:
Implements ARIMA for predictive analysis, providing trends with confidence intervals.
## Interactive Exploration:
Sidebar navigation allows users to explore world data, specific countries, top/bottom performers, and forecasts.

