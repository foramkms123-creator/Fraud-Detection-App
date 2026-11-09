import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳"
)

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter the transaction details below. "
    "The trained Random Forest model will classify the transaction "
    "as Fraud or Not Fraud."
)

# --------------------------------------------------
# Load trained artifacts
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model_RandomForest.joblib")
    scaler = joblib.load("scaler.joblib")
    selected_features = joblib.load("selected_features.joblib")
    encoder = joblib.load("partofday_encoder.joblib")

    return model, scaler, selected_features, encoder


model, scaler, selected_features, encoder = load_artifacts()


# --------------------------------------------------
# Input fields
# --------------------------------------------------

st.subheader("Transaction Details")

time_value = st.number_input(
    "Time",
    value=0.0,
    format="%.6f"
)

amount = st.number_input(
    "Amount",
    value=0.0,
    min_value=0.0,
    format="%.6f"
)

st.subheader("Transaction V Features")

input_data = {}

for i in range(1, 29):
    feature = f"V{i}"

    input_data[feature] = st.number_input(
        feature,
        value=0.0,
        format="%.6f"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Transaction"):

    try:

        # Create dataframe containing original features
        X = pd.DataFrame([input_data])

        X["Time"] = time_value
        X["Amount"] = amount

        # ------------------------------------------
        # Recreate PartOfDay feature
        # ------------------------------------------

        seconds_in_day = time_value % 86400
        hour = int(seconds_in_day // 3600)

        if 6 <= hour < 12:
            part_of_day = "morning"
        elif 12 <= hour < 18:
            part_of_day = "afternoon"
        elif 18 <= hour < 24:
            part_of_day = "evening"
        else:
            part_of_day = "night"

        X["PartOfDay"] = part_of_day

        # ------------------------------------------
        # Encode PartOfDay
        # ------------------------------------------

        encoded = encoder.transform(
            X[["PartOfDay"]]
        )

        encoded_columns = encoder.get_feature_names_out(
            ["PartOfDay"]
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoded_columns
        )

        # ------------------------------------------
        # Build same feature structure as training
        # ------------------------------------------

        X_encoded = pd.concat(
            [
                X.drop(
                    ["Time", "PartOfDay"],
                    axis=1
                ).reset_index(drop=True),

                encoded_df
            ],
            axis=1
        )

        # ------------------------------------------
        # Scale using saved scaler
        # ------------------------------------------

        X_scaled = scaler.transform(X_encoded)

        X_scaled_df = pd.DataFrame(
            X_scaled,
            columns=X_encoded.columns
        )

        # ------------------------------------------
        # Select exact 19 trained features
        # ------------------------------------------

        X_selected = X_scaled_df[selected_features]

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = model.predict(X_selected)[0]

        probabilities = model.predict_proba(
            X_selected
        )[0]

        fraud_probability = probabilities[1]
        not_fraud_probability = probabilities[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("🚨 FRAUD")

        else:

            st.success("✅ NOT FRAUD")

        st.subheader("Prediction Probabilities")

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

        st.subheader("Features Used by Model")

        st.dataframe(X_selected)

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )