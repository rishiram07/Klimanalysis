import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('environment.csv.csv',encoding='cp1252')
df = df.drop(['Area Code', 'Months Code', 'Element Code', 'Unit'], axis = 1)

df = df[df['Element'] == 'Temperature change']
df = df[df['Months'].isin(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])]
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace('y', '')

df1 = pd.melt(df, id_vars = ['area', 'months', 'element'], value_vars = ['1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'], var_name = 'years', value_name = 'temperature')

df['area'].unique()

world = df1.loc[df1.area == 'World']
world1 = world.groupby(['years']).mean()
# world1.head()

import plotly.express as px

def world_page():
    st.title("Temperature Change analysis and Prediction")
    st.subheader("World Analysis")
    fig = px.line(world1,  y='temperature', title='Average temperature change of the world from 1961 to 2019')
    st.plotly_chart(fig)
    # fig.show()

    fig = px.bar(world1,  y='temperature', title='Temperature of the world from 1961 to 2019')
    st.plotly_chart(fig)
    # fig.show()

    decades_df = world[world['years'].isin(['1961','1971','1981','1991','2001','2011'] )]
    # decades_df.head()

    fig = px.line(decades_df,x = "months", y="temperature", color='years', title='Comparing temperature of the world at start of every decade')
    st.plotly_chart(fig)
    # fig.show()

def algeria_page():
    st.subheader("Case Study : Algeria")

    Algeria = df1.loc[df1.area == 'Algeria']

    Algeria1 = Algeria.groupby(['years'], as_index = False).mean()
    # Algeria1.head()

    fig = px.bar(Algeria1, x="years", y="temperature", labels={'x':'Years', 'y':'Temperature'}, title='Tempereature in the Algeria from 1961 to 2019', color_discrete_map = {"temperature":"red"})
    st.plotly_chart(fig)
    # fig.show()

    Algeria_df = Algeria[Algeria['years'].isin(['1961', '1971', '1981', '1991', '2001', '2011'])]
    # Algeria_df.head()

    fig = px.line(Algeria_df,x = 'months', y="temperature", color='years', title='Comparing tempereature in Algeria at start of decade')
    st.plotly_chart(fig)
    # fig.show()

def france_page():
    st.subheader("Case Study : France")

    France = df1.loc[df1.area == 'France']

    France1 = France.groupby(['years'], as_index = False).mean()

    fig = px.bar(France1, x="years", y="temperature", labels={'x':'Years', 'y':'Temperature'}, title='Tempereature in the Algeria from 1961 to 2019', color_discrete_map = {"temperature":"red"})
    st.plotly_chart(fig)

    France_df = France[France['years'].isin(['1961', '1971', '1981', '1991', '2001', '2011'])]
    # France_df.head()

    fig = px.line(France_df,x = 'months', y="temperature", color='years', title='Comparing tempereature in france at start of decade')
    st.plotly_chart(fig)
    # fig.show()

def t10_b10():
    st.subheader("Finding top 10 and bottom 10 countries which have temperature change")

    df1 = pd.melt(df, id_vars = ['area', 'months', 'element'], value_vars = ['1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'], var_name = 'years', value_name = 'temperature')
    df1 = df1.groupby(['area', 'years']).mean().reset_index()

    df1['years'] = pd.to_datetime(df1['years'])
    df1 = df1.sort_values('years',ascending=True)
    df1['years'] = df1['years'].dt.strftime('%m/%d/%Y')

    top_10 = df1.groupby('area').sum().sort_values('temperature', ascending=False)[:10].reset_index()['area']
    # top_10

    bottom_10 = df1.groupby('area').sum().sort_values('temperature', ascending=True)[:10].reset_index()['area']
    # bottom_10

    countries = top_10.append(bottom_10)

    df1 = df1[df1['area'].isin(countries)]

    fig = px.bar(df1,x='temperature',y='area',animation_frame='years', hover_name='temperature', range_x=[-3.5,5.5], color='area')
    st.plotly_chart(fig)
    # fig.show()

def prediction():
    st.subheader("Simple Moving Average")

    world.plot(x = 'years', y = 'temperature')
    st.pyplot()
    # fig = px.line(x = world['years'], y = world['temperature'])
    # st.plotly_chart(fig)
    # fig.show()

    rolling_mean = world1['temperature'].rolling(window = 12).mean()
    rolling_std = world1['temperature'].rolling(window = 12).std()

    st.subheader("Rolling Mean & Rolling Standard Deviation")
    
    plt.plot(world1['temperature'], color = 'blue', label = 'Original')
    plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
    plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Rolling Standard Deviation')
    st.pyplot()
    # plt.show()

    from statsmodels.tsa.stattools import adfuller

    result = adfuller(world1['temperature'],autolag='AIC') 
    print('ADF Statistic: {}'.format(result[0])) 
    print('p-value: {}'.format(result[1]))  
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    st.subheader("AutoCorrelation")
    fig, ax = plt.subplots(figsize=(12,6))
    ax=plot_acf(world1['temperature'], ax)
    st.write(ax)

    st.subheader("Partial AutoCorrelation")
    ax=plot_pacf(world1['temperature'], lags = 10)
    st.write(ax)

    st.subheader("Auto Correlation Plot")

    from pandas.plotting import autocorrelation_plot

    autocorrelation_plot(world1['temperature'])

    def adf_test(series):
        result=adfuller(series)
        print('ADF Statistics: {}'.format(result[0]))
        print('p-value:{}'.format(result[1]))
        if result[1]<=0.05:
            print("Stationary")
        else:
            print("Not a stationary")

    adf_test(world1['temperature'].dropna())

    world1['difference']=world1['temperature']-world1['temperature'].shift(1)
    adf_test(world1['difference'].dropna())

    fig, axes = plt.subplots(2, 2, figsize=(20,10))
    axes[0, 0].plot(world1['temperature']); axes[0, 0].set_title('Original Series')
    plot_acf(world1['temperature'], ax=axes[0, 1])


    axes[1, 0].plot(world1['temperature'].diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(world1['temperature'].diff().dropna(), ax=axes[1, 1])
    st.pyplot()
    # plt.show()

    from statsmodels.tsa.arima.model import ARIMA

    mymodel = ARIMA(world1['temperature'], order = (1, 1, 2))  
    modelfit = mymodel.fit()  
    print(modelfit.summary())
    df1['forecast']=modelfit.predict(start=1,end=50,dynamic=True)
    df1[['temperature','forecast']].plot(figsize=(12,8))

    residuals = pd.DataFrame(modelfit.resid)
    fig, ax = plt.subplots(2,1, figsize=(7,11))

    st.subheader("Residuals and Density")

    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    st.pyplot()
    # plt.show()

def forecast_page():
    st.title("Forecast")
    st.image("forecast.png")

def data():
    st.subheader("Original Dataset")
    st.dataframe(df)
    st.write("data shape : ", df.shape)
    st.subheader("After preprocessing the Data")
    st.dataframe(df1)
    st.write("data shape : ", df1.shape)

def about_us():
    st.title("THANK YOU")
    st.subheader("About us")
    st.write("done by : ")
    st.write("Kartheepan G 20pd11")
    st.write("Mahitej K 20pd14")

page = {
    "World" : world_page,
    "Algeria" : algeria_page,
    "France" : france_page,
    "Top10 Bottom10" : t10_b10,
    "Prediction" : prediction,
    "Forecast" : forecast_page,
    "Data" : data,
    "About us" : about_us
}

pages = st.sidebar.selectbox("select the page :", page.keys())
page[pages]()