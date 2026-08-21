import joblib
import numpy as np
import pandas as pd
import sklearn as sk
import streamlit as st

st.set_page_config(page_title="House Rent Prediction", page_icon="🏠", layout="wide")

# ---------- Light custom styling ----------
st.markdown("""
    <style>
        .main > div { padding-top: 1.5rem; }
        h1 { font-weight: 700; }
        div[data-testid="stMetricValue"] { font-size: 2rem; color: #1a7f37; }
        .stButton > button {
            background-color: #1a7f37;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            height: 3em;
        }
        .stButton > button:hover {
            background-color: #156b2e;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# loadig the  Label Encoder Model
label_model = joblib.load('ml_Label_Encoder.pkl')

# loadig the  Label Regression Model
reg_model = joblib.load('ml_Regression_model.pkl')

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🏠 About")
    st.write(
        "This app estimates monthly house rent across major Indian cities "
        "using a Linear Regression model trained on real rental listings."
    )
    st.markdown("**Built by:** Aayla Zahid Butt")
    st.divider()
    st.caption("Model: Linear Regression (scikit-learn)")
    st.caption("Cities covered: Kolkata, Mumbai, Bangalore, Delhi, Chennai, Hyderabad")

# ---------- Header ----------
st.title("🏠 House Rent Prediction Model")
st.caption("Fill in the property details below to get an instant rent estimate.")
st.divider()

# ---------- Floor Details ----------
with st.container(border=True):
    st.subheader("🏢 Floor Details")
    col1, col2 = st.columns(2)
    with col1:
        floor_type = st.selectbox(
            "Floor Type",
            ["Ground", "Numbered Floor", "Upper Basement", "Lower Basement"]
        )
        if floor_type == "Numbered Floor":
            current_floor = st.number_input(
                "Which floor is the house on?",
                min_value=1, max_value=76, value=1, step=1
            )
            floor_label = str(int(current_floor))
        else:
            floor_label = floor_type
    with col2:
        total_floors = st.number_input(
            "Total floors in the building",
            min_value=1, max_value=89, value=1, step=1
        )

floor_str = f"{floor_label} out of {int(total_floors)}"

# ---------- Property Details ----------
with st.container(border=True):
    st.subheader("📐 Property Details")
    col3, col4, col5 = st.columns(3)
    with col3:
        BHK = st.number_input('Number of Bedrooms', min_value=0, value=None, step=1, placeholder="e.g. 2")
    with col4:
        Size = st.number_input('Size (sq ft)', min_value=0, value=None, step=10, placeholder="e.g. 900")
    with col5:
        Bathroom = st.number_input('Number of Bathrooms', min_value=0, value=None, step=1, placeholder="e.g. 2")

# ---------- Location & Preferences ----------
with st.container(border=True):
    st.subheader("📍 Location & Preferences")
    col6, col7 = st.columns(2)
    with col6:
        Area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])
        City = st.selectbox("City", ["Kolkata", "Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"])
        Furnish_Status = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
    with col7:
        Point_of_Contact = st.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"])
        Tenant_Preferred = st.selectbox("Tenant Preferred", ["Bachelors", "Bachelors/Family", "Family"])
        Area_Locality = st.text_input('Area Locality', placeholder="e.g. Bandra West, Gachibowli")

st.write("")
predict_clicked = st.button('🔍 Predict Rent', use_container_width=True)

# --- Predict button ---
if predict_clicked:
    if BHK is None or Size is None or Bathroom is None or not Area_Locality:
        st.warning("Please fill in all fields before predicting.")
    else:
        try:
            # Encode each categorical field using its own saved encoder
            floor_enc = label_model['Floor'].transform([floor_str])[0]
            area_type_enc = label_model['Area Type'].transform([Area_type])[0]
            city_enc = label_model['City'].transform([City])[0]
            furnish_enc = label_model['Furnishing Status'].transform([Furnish_Status])[0]
            tenant_enc = label_model['Tenant Preferred'].transform([Tenant_Preferred])[0]
            poc_enc = label_model['Point of Contact'].transform([Point_of_Contact])[0]

            # Area Locality might contain a value the encoder never saw during training
            locality_encoder = label_model['Area Locality']
            if Area_Locality in locality_encoder.classes_:
                locality_enc = locality_encoder.transform([Area_Locality])[0]
            else:
                st.warning(
                    f"'{Area_Locality}' wasn't in the training data — "
                    "using the most common locality as a fallback."
                )
                locality_enc = 0

            features = np.array([[
                BHK, Size, floor_enc, area_type_enc, locality_enc,
                city_enc, furnish_enc, tenant_enc, Bathroom, poc_enc
            ]])

            prediction = reg_model.predict(features)[0]

            st.divider()
            result_col1, result_col2 = st.columns([2, 1])
            with result_col1:
                st.metric(label="Predicted Monthly Rent", value=f"₹{prediction:,.0f}")
                st.success("Prediction generated successfully.")
            with result_col2:
                st.markdown("**Summary**")
                st.write(f"📍 {City}, {Area_Locality}")
                st.write(f"🛏️ {int(BHK)} BHK · 🚿 {int(Bathroom)} Bath · 📐 {int(Size)} sq ft")
                st.write(f"🏢 {floor_str} · {Furnish_Status}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
