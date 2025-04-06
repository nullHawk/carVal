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

def model_pred(
        fuel_type,
        transmission_type,
        engine,
        seats,
        year=2018,
        range=4000):
    reg_model = load_model()
    input_features = [
        [
            float(year),
            1,
            range, 
            fuel_type,
            transmission_type,
            19.70,
            engine,
            86.30,
            seats
        ]
    ]
    return reg_model.predict(input_features)

# Use a form to group widgets and control execution
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    fuel_type = col1.selectbox("Select the fuel type",
                               ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
    
    year = col1.number_input("Enter the year of manufacture",
                             min_value=1900, max_value=2023, value=2018, step=1)

    engine = col1.slider("Set the Engine Power",
                         500, 5000, step=100)

    transmission_type = col2.selectbox("Select the transmission type",
                                       ["Manual", "Automatic"])
    
    range = col2.slider("Select KMs Driven",
                        0, 200000, step=1000)

    seats = col2.selectbox("Enter the number of seats",
                           [4, 5, 7, 9, 11])

    # Submit button for the form
    submit_button = st.form_submit_button("Predict Price")

# Use session state to control when the prediction is triggered
if "predicted_price" not in st.session_state:
    st.session_state.predicted_price = None

if submit_button:
    fuel_type_encoded = encode_dict['fuel_type'][fuel_type]
    transmission_type_encoded = encode_dict['transmission_type'][transmission_type]

    # Perform prediction and store it in session state
    st.session_state.predicted_price = model_pred(fuel_type_encoded, transmission_type_encoded, engine, seats, year, range)[0]

# Display the predicted price if available
if st.session_state.predicted_price is not None:
    st.text(f"Predicted Price of the car is: {st.session_state.predicted_price:.2f} Lakhs")