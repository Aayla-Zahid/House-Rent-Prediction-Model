import numpy as np
import pandas as pd
import sklearn as sk
import streamlit as st
import joblib

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

# loading the dataset so the Floor and Area Locality dropdowns always show
# every unique value that exists in the training data
data_df = pd.read_csv('House_Rent_Dataset.csv')
floor_options = sorted(data_df['Floor'].dropna().unique().tolist())
area_locality_options = sorted(data_df['Area Locality'].dropna().unique().tolist())


def safe_encode(encoder, value, field_name):
    """
    Encode a categorical value using a fitted LabelEncoder.
    If the value was never seen during training, fall back to the
    encoder's first known class instead of crashing, and warn the user.
    """
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        st.warning(
            f"'{value}' wasn't in the training data for '{field_name}' — "
            f"using '{encoder.classes_[0]}' as a fallback."
        )
        return encoder.transform([encoder.classes_[0]])[0]


def safe_encode_floor(encoder, floor_label, total_floors, field_name="Floor"):
    """
    Encode the 'Floor' field ('<floor_label> out of <total_floors>').
    If the exact string wasn't seen in training, search the encoder's
    known classes for the same floor_label and pick the one whose total
    floor count is numerically closest, instead of falling back blindly.
    """
    exact = f"{floor_label} out of {int(total_floors)}"
    if exact in encoder.classes_:
        return encoder.transform([exact])[0]

    # Look for other known classes with the same floor label
    candidates = []
    for cls in encoder.classes_:
        if cls.startswith(f"{floor_label} out of "):
            try:
                cls_total = int(cls.split("out of")[-1].strip())
                candidates.append((abs(cls_total - int(total_floors)), cls))
            except ValueError:
                continue

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_match = candidates[0][1]
        st.warning(
            f"'{exact}' wasn't in the training data — "
            f"using the closest known match '{best_match}' instead."
        )
        return encoder.transform([best_match])[0]

    # No match at all for this floor label — fall back to the first known class overall
    st.warning(
        f"'{floor_label}' floor type wasn't in the training data at all — "
        f"using '{encoder.classes_[0]}' as a fallback. Prediction may be less accurate."
    )
    return encoder.transform([encoder.classes_[0]])[0]


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
    floor_str = st.selectbox("Floor", floor_options)

# ---------- Property Details ----------
with st.container(border=True):
    st.subheader("📐 Property Details")
    col3, col4, col5 = st.columns(3)
    with col3:
        BHK = st.number_input('BHK', min_value=0, value=None, step=1, placeholder="e.g. 2")
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
        Area_Locality = st.selectbox("Area Locality", area_locality_options)

st.write("")
predict_clicked = st.button('🔍 Predict Rent', use_container_width=True)

# --- Predict button ---
if predict_clicked:
    if BHK is None or Size is None or Bathroom is None or not Area_Locality:
        st.warning("Please fill in all fields before predicting.")
    else:
        try:
            # Encode each categorical field using its own saved encoder,
            # safely falling back to a known class if the value is unseen
            floor_enc = safe_encode(label_model['Floor'], floor_str, "Floor")
            area_type_enc = safe_encode(label_model['Area Type'], Area_type, "Area Type")
            city_enc = safe_encode(label_model['City'], City, "City")
            furnish_enc = safe_encode(label_model['Furnishing Status'], Furnish_Status, "Furnishing Status")
            tenant_enc = safe_encode(label_model['Tenant Preferred'], Tenant_Preferred, "Tenant Preferred")
            poc_enc = safe_encode(label_model['Point of Contact'], Point_of_Contact, "Point of Contact")
            locality_enc = safe_encode(label_model['Area Locality'], Area_Locality, "Area Locality")

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
