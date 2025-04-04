import pandas as pd
import streamlit as st
import pickle

cars_df = pd.read_excel("data/cars24-car-price.xlsx")

st.write(
    """
     # Car Price Prediction
    """
)
st.dataframe(cars_df.head())

encode_dict = {
    "fuel_type": {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5},
    "seller_type": {'Dealer': 1, 'Individual': 2, 'Trustmark Dealer': 3},
    "transmission_type": {'Manual': 1, 'Automatic': 2}
}

@st.cache_resource
def load_model():
    with open("model/car_pred", "rb") as file:
        return pickle.load(file)

def model_pred(fuel_type, transmission_type, engine, seats):
    reg_model = load_model()
    input_features = [[2018.0, 1, 4000, fuel_type, transmission_type, 19.70, engine, 86.30, seats]]
    return reg_model.predict(input_features)

## Formatting and adding dropdowns and sliders
col1, col2 = st.columns(2)

fuel_type = col1.selectbox("Select the fuel type",
                           ["Diesel", "Petrol", "CNG", "LPG", "Electric"])

engine = col1.slider("Set the Engine Power",
                     500, 5000, step=100)

transmission_type = col2.selectbox("Select the transmission type",
                                   ["Manual", "Automatic"])

seats = col2.selectbox("Enter the number of seats",
                       [4,5,7,9,11])

if (st.button("Predict Price")):
    fuel_type = encode_dict['fuel_type'][fuel_type]
    transmission_type = encode_dict['transmission_type'][transmission_type]

    price = model_pred(fuel_type, transmission_type, engine, seats)
    st.text("Predicted Price of the car is: " + str(price))
