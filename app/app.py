import os

import mlflow
import pandas as pd
import requests
import streamlit as st
from mlflow.pyfunc import PyFuncModel
from requests.auth import HTTPBasicAuth


# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="Hotel Reservations ML Inference",
    page_icon="🏡",
    layout="wide",
)


# Specify your Unity Catalog model path (update with your actual catalog, schema, model, and alias/version)
MODEL_URI = "models:/mlops_dev.olalubic.hotel_reservations_model_custom@latest-model"

# Trick to ensure DATABRICKS_HOST is set with 'https://' prefix."""
raw_host = os.environ["DATABRICKS_HOST"]
host = raw_host if raw_host.startswith("https://") else f"https://{raw_host}"


def get_token()-> str:
    """
    Retrieves an OAuth access token from the Databricks workspace.

    :return: The access token string.
    """    
    response = requests.post(
        f"{host}/oidc/v1/token",
        auth=HTTPBasicAuth(
            os.environ["DATABRICKS_CLIENT_ID"], 
            os.environ["DATABRICKS_CLIENT_SECRET"]
            ),
        data={
            'grant_type': 'client_credentials',
             'scope': 'all-apis'
             }
    )

    return response.json()["access_token"]

os.environ["DATABRICKS_TOKEN"] = get_token()

# Set MLflow to use Unity Catalog as the registry
mlflow.set_registry_uri("databricks-uc")


@st.cache_resource
def load_uc_model()-> PyFuncModel:
    """
    Loads a PyFunc model from the specified MLflow model URI.

    :return: The loaded MLflow PyFuncModel.
    """    
    return mlflow.pyfunc.load_model(MODEL_URI)


model = load_uc_model()

# --- SIDEBAR ---
with st.sidebar:
     st.image("./hotel.png", width=300)
     st.title("🏡 Hotel Reservations Predictor")
     st.markdown("This app predicts house prices using a Databricks Unity Catalog ML model.")
     st.markdown("**Instructions:**\n- Fill in the property details below\n- Click 'Predict' to get the estimated price")

st.title("ML Inference with Unity Catalog Model (Databricks Apps)")

# --- LAYOUT: MAIN INPUTS IN COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    no_of_adults = st.number_input("no of adults", value=1, min_value=1, step=1)
    no_of_children     = st.number_input("no of children", value=0, min_value=0, step=1)
    no_of_weekend_nights = st.number_input("no of weekend nights", value=1, min_value=0, max_value=10, step=1)
    no_of_week_nights = st.number_input("no of week nights", value=0, min_value=0, max_value=10)
    lead_time   = st.number_input("lead time", min_value=0, max_value=100, value=10)

with col2:
    repeated_guest = st.number_input("repeated guest", min_value=0, max_value=1, value=1)
    no_of_previous_cancellations   = st.number_input("no of previous cancellations", min_value=0, value=100)
    no_of_previous_bookings_not_canceled  = st.number_input("no of previous bookings not canceled", min_value=0, value=100)
    avg_price_per_room    = st.number_input("avg price per room", min_value=10.0, value=1000.0)
    no_of_special_requests   = st.number_input("no of special requests", min_value=0, max_value=10, value=0)

with col3:
    required_car_parking_space = st.number_input("required car parking space", min_value=0, max_value=3, value=1)
    room_type_reserved = st.selectbox("room type reserved", options=['Room_Type 1', 'Room_Type 4'])
    market_segment_type = st.selectbox("market segment type", options=['Online', 'Offline'])
    country = st.selectbox("country", options=['PL', 'UK'])
    arrival_month = st.number_input("arrival month", min_value=1, max_value=12, value=1)

# --- EXPANDER FOR ADVANCED/CATEGORICAL FIELDS ---
with st.expander("Show More Property Details", expanded=False):
    type_of_meal_plan = st.selectbox("type of meal plan", options=['Not Selected', 'Meal Plan 1', 'Meal Plan 2'])

# --- DATAFRAME PREPARATION ---
input_df = pd.DataFrame([[
    no_of_adults,
    no_of_children,
    no_of_weekend_nights,
    no_of_week_nights,
    lead_time,
    repeated_guest,
    no_of_previous_cancellations,
    no_of_previous_bookings_not_canceled,
    avg_price_per_room,
    no_of_special_requests,
    required_car_parking_space,
    room_type_reserved,
    market_segment_type,
    country,
    arrival_month,
    type_of_meal_plan,
    "INN25630",
    "1sw2221"
]], columns=[
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "lead_time",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "required_car_parking_space",
    "room_type_reserved",
    "market_segment_type",
    "country",
    "arrival_month",
    "type_of_meal_plan",
    "Booking_ID",
    "Client_ID"
])

# --- PREDICTION BUTTON ---
st.markdown("---")
if st.button("🔮 Predict Hotel Reservation"):
    prediction = model.predict(input_df)
    st.success(f"🏷️ Prediction: ${prediction[0]:,}")

