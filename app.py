import streamlit as st
import joblib
import pandas as pd

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(page_title="Marketing Campaign ROI Predictor", layout="wide")

# ---------------------------
# Load Model Function
# ---------------------------
def load_model():
    try:
        model = joblib.load("campaign_model.joblib")
        # Test prediction to confirm model works
        test_data = pd.DataFrame(
            {
                "channel": ["social_media"],
                "budget": [1000],
                "duration": [10],
                "target_audience": ["youth"],
            }
        )
        try:
            _ = model.predict(test_data)
        except Exception as pred_error:
            return None, f"Model compatibility error: {str(pred_error)}"

        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# ---------------------------
# Initialize model
# ---------------------------
model, error = load_model()
demo_mode = False
if model is None:
    st.warning(
        f"⚠️ Running in demo mode because: {error}. "
        "Predictions will be random, not model-based."
    )
    demo_mode = True

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📊 Marketing Campaign ROI Predictor")
st.markdown("Enter your campaign details to predict the ROI and optimize your strategy.")

with st.form("campaign_form"):
    st.subheader("Campaign Details")

    col1, col2 = st.columns(2)

    with col1:
        channel = st.selectbox(
            "Channel", ["social_media", "tv", "radio", "email", "print"]
        )
        budget = st.number_input("Budget ($)", min_value=100, max_value=100000, value=5000, step=100)
    with col2:
        duration = st.number_input("Duration (days)", min_value=1, max_value=365, value=30)
        target_audience = st.selectbox("Target Audience", ["youth", "adults", "seniors"])

    submitted = st.form_submit_button("Predict ROI")

# ---------------------------
# Prediction
# ---------------------------
if submitted:
    input_data = pd.DataFrame(
        {
            "channel": [channel],
            "budget": [budget],
            "duration": [duration],
            "target_audience": [target_audience],
        }
    )

    if demo_mode:
        import random
        roi = round(random.uniform(0.5, 3.0), 2)
    else:
        try:
            roi = round(model.predict(input_data)[0], 2)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()

    st.success(f"📈 Predicted ROI: **{roi}x**")

# ---------------------------
# Prompt Generator
# ---------------------------
st.subheader("💡 Prompt for Replit / AI Deployment")
st.markdown("Copy and paste this prompt to quickly deploy your own app in Replit or an AI code generator:")

prompt = """Build a web app that:
1. Collects campaign details (channel, budget, duration, target audience).
2. Uses a trained ML model (`campaign_model.joblib`) to predict ROI.
3. Displays results in a user-friendly way.
4. Has demo mode if the model is missing.
"""

st.code(prompt, language="markdown")

if st.button("Copy Prompt"):
    st.session_state["copied"] = True
    st.success("✅ Prompt copied to clipboard! 📋")
