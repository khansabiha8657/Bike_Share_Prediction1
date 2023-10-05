import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('new_data.csv')
x = df.drop(['Rented Bike Count'], axis=1)
y = df['Rented Bike Count'].values
regr = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
regr.fit(X_train, y_train)

st.set_page_config(
    page_title="Seoul Bike Sharing Demand Prediction",
    page_icon="🚴‍♂️",  # Replace with your own icon
    layout="wide"
)

# Create a title for app
title_html = "<h1 style='text-align: center;'>&nbsp; Seoul Bike Sharing Demand Prediction🚴</h1>"
st.markdown(title_html, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("About This App")
st.sidebar.markdown("This app predicts bike rental counts in Seoul, South Korea based on input features.")
st.sidebar.markdown("The prediction model is based on a Random Forest Regressor with an 84% R-squared score.")
st.sidebar.markdown("---")

# Slider default values
default_hour = 12
default_temperature = 15
default_humidity = 50
default_windspeed = 3.0
default_position = 1000
default_dew = 10
default_solar = 1.0
default_rainfall = 5
default_snowfall = 0.0
default_seasons = 2
default_holiday = 0
default_functioningday = 1

hour = st.slider("Hour of the Day (0-23)", 0, 23, default_hour)
temperature = st.slider("Temperature (°C)", -25, 25, default_temperature)
humidity = st.slider("Humidity (%)", 0, 100, default_humidity)
windspeed = st.slider("Windspeed (m/s)", 0.0, 7.4, default_windspeed)
position = st.slider("Visibility (m)", 0, 2000, default_position)
dew = st.slider("Dew Point (°C)", -40, 30, default_dew)
solar = st.slider("Solar Radiation (MJ/m^2)", 0.0, 4.0, default_solar)
rainfall = st.slider("Rainfall (mm)", 0, 100, default_rainfall)
snowfall = st.slider("Snowfall (cm)", 0.0, 20.0, default_snowfall)
seasons = st.slider("Seasons (0: Spring, 1: Summer, 2: Fall, 3: Winter)", 0, 3, default_seasons)
holiday = st.selectbox("Is it a Holiday?", ["No", "Yes"], index=default_holiday)
functioningday = st.selectbox("Is it a Functioning Day?", ["No", "Yes"], index=default_functioningday)

# Convert holiday and functioningday to 0 or 1
holiday = 1 if holiday == "Yes" else 0
functioningday = 1 if functioningday == "Yes" else 0

inputs = [[hour, temperature, humidity, windspeed, position, dew, solar, rainfall, snowfall, seasons, holiday, functioningday]]

if st.button('Predict'):
    result = regr.predict(inputs)
    updated_result = result.flatten().astype(int)
    
    # Display a dynamic icon and color based on the predicted value
    if updated_result >= 0:
        icon_html = "🚴‍♀"  # Sun icon for positive prediction
        icon_color = "green"
    else:
        icon_html = "🚴‍♀"  # Rain icon for negative prediction
        icon_color = "red"
    
    # Style the predicted value with an icon and color
    st.subheader("Predicted Bike Rental Count:")
    st.markdown(f"<p style='color: {icon_color}; font-size: 24px;'>{icon_html} {updated_result}</p>", unsafe_allow_html=True)
