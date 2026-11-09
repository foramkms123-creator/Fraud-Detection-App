import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("model_v2.joblib")
    scaler = joblib.load("scaler_v2.joblib")
    features = joblib.load("features_v2.joblib")

    return model, scaler, features


model, scaler, features = load_artifacts()


# ============================================================
# LOAD DATASET FOR TESTING
# ============================================================

dataset = pd.read_csv("../data/sampledcreditcard.csv")


# ============================================================
# TITLE
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter the transaction details below and the trained "
    "machine learning model will predict whether the transaction "
    "is potentially fraudulent."
)

st.info(
    "Note: V1–V28 are anonymized numerical features from the "
    "credit-card fraud dataset. Their real-world meanings are "
    "not provided by the dataset."
)


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.header("Transaction Details")


# ============================================================
# TEST USING DATASET ROW
# ============================================================

st.subheader("🧪 Test Using Dataset")

st.write(
    f"Dataset contains {len(dataset)} transactions."
)

row_number = st.number_input(
    "Select Dataset Row",
    min_value=1,
    max_value=len(dataset),
    value=1,
    step=1
)


if st.button("📥 Load Selected Transaction"):

    selected_row = dataset.iloc[row_number - 1]

    st.session_state["loaded_row"] = selected_row.to_dict()

    st.success(
        f"Transaction row {row_number} loaded successfully."
    )


# ============================================================
# TIME AND AMOUNT
# ============================================================

col1, col2 = st.columns(2)


with col1:

    time_value = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=float(
            st.session_state.get(
                "loaded_row",
                {}
            ).get("Time", 0.0)
        ),
        format="%.2f"
    )


with col2:

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=float(
            st.session_state.get(
                "loaded_row",
                {}
            ).get("Amount", 100.0)
        ),
        format="%.2f"
    )


# ============================================================
# V1 - V28 FEATURES
# ============================================================

st.header("Dataset Features")

st.caption(
    "These V1–V28 values are anonymized features used by "
    "the original dataset."
)


input_data = {}

columns = st.columns(4)


for i in range(1, 29):

    feature = f"V{i}"

    with columns[(i - 1) % 4]:

        input_data[feature] = st.number_input(
            feature,
            value=float(
                st.session_state.get(
                    "loaded_row",
                    {}
                ).get(feature, 0.0)
            ),
            format="%.6f"
        )


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔍 Predict Transaction", type="primary"):

    try:

        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        X = pd.DataFrame([input_data])

        X["Time"] = time_value
        X["Amount"] = amount


        # ----------------------------------------------------
        # Create PartOfDay
        # ----------------------------------------------------

        seconds_in_day = time_value % 86400

        hour = int(
            seconds_in_day // 3600
        )


        if 6 <= hour < 12:

            part_of_day = "morning"

        elif 12 <= hour < 18:

            part_of_day = "afternoon"

        elif 18 <= hour < 24:

            part_of_day = "evening"

        else:

            part_of_day = "night"


        # ----------------------------------------------------
        # One-hot encode PartOfDay
        # ----------------------------------------------------

        X["PartOfDay"] = part_of_day

        X = pd.get_dummies(
            X,
            columns=["PartOfDay"],
            drop_first=True
        )


        # ----------------------------------------------------
        # Make sure all expected columns exist
        # ----------------------------------------------------

        for column in [
            "PartOfDay_evening",
            "PartOfDay_morning",
            "PartOfDay_night"
        ]:

            if column not in X.columns:

                X[column] = 0


        # ----------------------------------------------------
        # Arrange columns in exact training order
        # ----------------------------------------------------

        X = X[features]


        # ----------------------------------------------------
        # Scale features
        # ----------------------------------------------------

        X_scaled = scaler.transform(X)


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            X_scaled
        )[0]

        probabilities = model.predict_proba(
            X_scaled
        )[0]


        not_fraud_probability = probabilities[0]

        fraud_probability = probabilities[1]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.header("Prediction Result")


        if prediction == 1:

            st.error(
                "🚨 FRAUD DETECTED"
            )

        else:

            st.success(
                "✅ NOT FRAUD"
            )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.subheader(
            "Prediction Probability"
        )

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Not Fraud",
                f"{not_fraud_probability * 100:.2f}%"
            )


        with col2:

            st.metric(
                "Fraud",
                f"{fraud_probability * 100:.2f}%"
            )


        # ====================================================
        # TRANSACTION INFORMATION
        # ====================================================

        st.subheader(
            "Transaction Information"
        )

        st.write(
            f"**Amount:** ₹{amount:.2f}"
        )

        st.write(
            f"**Time:** {time_value:.2f}"
        )

        st.write(
            f"**Part of Day:** {part_of_day.title()}"
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )