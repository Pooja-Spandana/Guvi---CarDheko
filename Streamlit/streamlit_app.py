# Importing libraries
import os
import pickle
import pandas as pd
import streamlit as st

# Get absolute path to repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up (since streamlit_app.py is inside Streamlit/)
ROOT_DIR = os.path.dirname(BASE_DIR)

# Build safe paths
MODEL_PATH = os.path.join(ROOT_DIR, "saved_pickles", "final_model.pkl")
ENCODER_PATH = os.path.join(ROOT_DIR, "saved_pickles", "label_encoders.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "saved_pickles", "mm_scaler.pkl")

# Load pickles
with open(MODEL_PATH, "rb") as f:
    loaded_model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    loaded_label_encoders = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    loaded_mm_scaler = pickle.load(f)

# Load dataset
df = pd.read_csv("All_Cities_Cleaned.csv")

# Get min & max values dynamically
min_kms, max_kms = df["Kms Driven"].min(), df["Kms Driven"].max()
min_mileage, max_mileage = df["Mileage"].min(), df["Mileage"].max()
min_engine, max_engine = df["Engine"].min(), df["Engine"].max()
min_power, max_power = df["Max Power"].min(), df["Max Power"].max()
min_age, max_age = df["Car Age"].min(), df["Car Age"].max()

# ---- Initialize Page State ----
if "page" not in st.session_state:
    st.session_state["page"] = "Home"
if "predicted_price" not in st.session_state:
    st.session_state["predicted_price"] = None  # Stores the predicted price
if "form_data" not in st.session_state:
    st.session_state["form_data"] = {}  # Stores user input data

# ---- Sidebar Content Based on Active Page ----
if st.session_state["page"] == "Home":
    st.sidebar.markdown("<h1 style='text-align: left;'>✇ USED CAR PRICE PREDICTOR</h1>", unsafe_allow_html=True)

    st.sidebar.subheader("Home page")
    st.sidebar.info("""
        - This website uses a **Machine Learning** model trained on historical car price data.
        - The model predicts used car prices with **~95.35% accuracy**.
        - The model is trained on real-world data sourced from **Cardheko** .
    """)

    st.sidebar.subheader("Contact Us:")
    st.sidebar.info("""
        - India
        - +91 7853498267
        - carpricepredictor@support.com
    """)

    if st.sidebar.button("**Get Started ➡️**", use_container_width=True):
        st.session_state["page"] = "Prediction"
        st.rerun()

elif st.session_state["page"] == "Prediction":
    st.sidebar.markdown("<h1 style='text-align: left;'>✇ USED CAR PRICE PREDICTOR</h1>", unsafe_allow_html=True)
    st.sidebar.subheader("User Guide")
    st.sidebar.info("""
    - Fill in your specific car requirments to get a price estimate.
    - If any requirement is not specified the default values will be used to get an estimate.
    - Click 'Predict Price' to see the results.
    """)

    st.sidebar.subheader("Contact Us:")
    st.sidebar.info("""
        - India
        - +91 7853498267
        - carpricepredictor@support.com
    """)

elif st.session_state["page"] == "Result":
    st.sidebar.markdown("<h1 style='text-align: left;'>✇ USED CAR PRICE PREDICTOR</h1>", unsafe_allow_html=True)
    st.sidebar.subheader("User Guide")
    st.sidebar.info("""
    - **Note:** The predicted price is only an estimate, actual price may vary 
    - Click 'Try Another Estimate' to get a new estimate.
    """)

    st.sidebar.subheader("Contact Us:")
    st.sidebar.info("""
        - India
        - +91 7853498267
        - carpricepredictor@support.com
    """)

    st.sidebar.markdown("<h2 style='text-align: center;'>THANK YOU, VISIT US AGAIN!</h2>", unsafe_allow_html=True)

# ---- Home Page ----
if st.session_state["page"] == "Home":
    st.image(r"C:\Users\spand\Projects\CAR_DHEKO\Streamlit\Images\1.jpg")
    
# ---- Prediction Page ----
elif st.session_state["page"] == "Prediction":  
    # Display form
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("<h1 style='text-align: center;'>ENTER CAR DETAILS</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            city = st.selectbox("🏙️ City", df["City"].unique(), key="city")
            fuel_type = st.selectbox("⛽ Fuel Type", df["Fuel Type"].unique(), key="fuel_type")
            brand = st.selectbox("🏢 Brand", df["Brand"].unique(), key="brand")
            kms_driven = st.slider("🚗 Kms Driven", min_value=min_kms, max_value=max_kms, step=500, key="kms_driven")
            mileage = st.slider("📏 Mileage (km/l)", min_value=min_mileage, max_value=max_mileage, step=1, key="mileage")
        
        with col2:
            transmission = st.selectbox("⚙️ Transmission", df["Transmission"].unique(), key="transmission")
            owner_number = st.selectbox("👤 Owner Number", sorted(df["Owner Number"].unique()), key="owner_number")
            car_age = st.selectbox("📅 Car Age (years)", options=list(range(min_age, max_age + 1)), key="car_age")
            engine = st.slider("🔧 Engine Capacity (CC)", min_value=min_engine, max_value=max_engine, step=100, key="engine")
            max_power = st.slider("⚡ Max Power (bhp)", min_value=min_power, max_value=max_power, step=5, key="max_power")

        col1, col2, col3 = st.columns([10, 200, 10])  # Adjust width ratio
        with col2:
            if st.form_submit_button("**Predict Price 💰**",  use_container_width=True):
                st.session_state["page"] = "Result"
                st.session_state["form_data"] = {
                    "city": city, "fuel_type": fuel_type, "brand": brand, "kms_driven": kms_driven,
                    "mileage": mileage, "transmission": transmission, "owner_number": owner_number,
                    "car_age": car_age, "engine": engine, "max_power": max_power
                }
    
                # Feature Engineering
                age_vs_performance = car_age * mileage
                fuel_efficiency = mileage / engine if engine != 0 else 0  

                # Encode categorical features
                transmission_encoded = loaded_label_encoders["Transmission"].transform([transmission])[0]
                brand_encoded = loaded_label_encoders["Brand"].transform([brand])[0]
                fuel_type_encoded = loaded_label_encoders["Fuel Type"].transform([fuel_type])[0]
                city_encoded = loaded_label_encoders["City"].transform([city])[0]

                # Create DataFrame for Model
                input_data = pd.DataFrame([[city_encoded, kms_driven, transmission_encoded, owner_number, brand_encoded, mileage,
                                            engine, max_power, car_age, age_vs_performance, fuel_efficiency]],
                                        columns=['City','Kms Driven', 'Transmission', 'Owner Number', 'Brand', 'Mileage',
                                                'Engine', 'Max Power', 'Car Age', 'Age vs Performance', 'Fuel Efficiency'])

                # Scale Numerical Features
                input_data[["Kms Driven", "Mileage", "Engine", "Max Power", "Car Age", "Age vs Performance", "Fuel Efficiency"]] = loaded_mm_scaler.transform(
                    input_data[["Kms Driven", "Mileage", "Engine", "Max Power", "Car Age", "Age vs Performance", "Fuel Efficiency"]])

                st.session_state["predicted_price"] = loaded_model.predict(input_data)[0]
                st.rerun()

# ---- Result Page ----
elif st.session_state["page"] == "Result":
    with st.form("result_form"):
        st.markdown("<h2 style='text-align: center;'>PREDICTED PRICE</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(r"C:\Users\spand\Projects\CAR_DHEKO\Streamlit\Images\2.jpg", use_container_width=True)

        with col2:
            st.markdown("<h4 style='text-align: center;'>Entered Car Details</h4>", unsafe_allow_html=True)
            for key, value in st.session_state["form_data"].items():
                st.markdown(f"<p style='text-align: center; font-size:18px; font-weight:bold;'>{key.replace('_', ' ').title()}: {value}</p>", unsafe_allow_html=True)
        
        with col3:
            st.image(r"C:\Users\spand\Projects\CAR_DHEKO\Streamlit\Images\3.jpg", use_container_width=True)

        st.markdown(f"""
            <div style='background-color:#366788; padding:15px; border-radius:10px; text-align:center; font-size:24px;'>
                Estimated Price: <strong>$ {st.session_state["predicted_price"]:,.2f}</strong>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([10, 200, 10])  # Adjust width ratio
        with col2:
            if st.form_submit_button("**Try Another Estimate 🔄**", use_container_width=True):
                st.session_state["page"] = "Prediction"
                st.session_state["form_data"] = {}  # Reset form data
                st.session_state["predicted_price"] = None  # Reset prediction
                st.rerun()