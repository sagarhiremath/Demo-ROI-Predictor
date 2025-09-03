import streamlit as st
import pandas as pd
import joblib
import os
from typing import Dict, Tuple

# Page configuration
st.set_page_config(
    page_title="Marketing Campaign ROI Predictor",
    page_icon="📈",
    layout="wide"
)

def load_model():
    """Load the pre-trained scikit-learn model"""
    try:
        if os.path.exists('campaign_model.joblib'):
            model = joblib.load('campaign_model.joblib')
            return model, None
        else:
            return None, "Model file 'campaign_model.joblib' not found. Please ensure the model file is uploaded to the application directory."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def get_roi_category_and_recommendation(roi: float) -> Tuple[str, str, str]:
    """Categorize ROI and provide recommendations"""
    if roi > 1.0:
        category = "High"
        color = "green"
        recommendation = "Scale budget by 10–25%."
    elif roi >= 0.3:
        category = "Medium" 
        color = "orange"
        recommendation = "A/B test creatives and adjust targeting."
    else:
        category = "Low"
        color = "red"
        recommendation = "Change target audience or optimize creative."
    
    return category, color, recommendation

def create_genai_prompt(campaign_data: Dict) -> str:
    """Generate a sample GenAI prompt for campaign optimization"""
    prompt = f"""Campaign: {campaign_data['campaign_name']}
Type: {campaign_data['campaign_type']}
Audience: {campaign_data['audience']}
Cost: ${campaign_data['cost']:,.2f}
Predicted Revenue: ${campaign_data['predicted_revenue']:,.2f}
ROI: {campaign_data['roi']:.2f}

Write a short summary and 3 actionable recommendations."""
    
    return prompt

def main():
    # Title and header
    st.title("📈 Marketing Campaign ROI Predictor (Demo)")
    st.markdown("Predict marketing campaign revenue and analyze ROI performance with actionable recommendations.")
    
    # Load model
    model, error_message = load_model()
    
    if error_message:
        st.error(error_message)
        st.info("Please upload the 'campaign_model.joblib' file to use this application.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Campaign Input Form")
        
        # Create form for inputs
        with st.form("campaign_form"):
            # Campaign Name
            campaign_name = st.text_input(
                "Campaign Name",
                placeholder="Enter campaign name...",
                help="A descriptive name for your marketing campaign"
            )
            
            # Campaign Type dropdown
            campaign_type = st.selectbox(
                "Campaign Type",
                options=['Social Media', 'Email', 'Search', 'Display', 'Referral'],
                help="Select the type of marketing campaign"
            )
            
            # Audience dropdown
            audience = st.selectbox(
                "Audience",
                options=['New Customers', 'Existing Customers', 'Lookalike', 'High Value', 'All'],
                help="Target audience for the campaign"
            )
            
            # Cost input
            cost = st.number_input(
                "Cost ($)",
                min_value=0.0,
                value=1000.0,
                step=50.0,
                help="Total campaign budget in dollars"
            )
            
            # Average Order Value input
            avg_order_value = st.number_input(
                "Average Order Value ($)",
                min_value=0.01,
                value=75.0,
                step=5.0,
                help="Expected average order value per customer"
            )
            
            # Submit button
            submitted = st.form_submit_button("🚀 Predict Revenue & ROI", use_container_width=True)
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        if submitted:
            if not campaign_name.strip():
                st.warning("Please enter a campaign name.")
                return
            
            try:
                # Prepare data for prediction
                input_data = pd.DataFrame({
                    'Campaign_Type': [campaign_type],
                    'Audience': [audience],
                    'Cost': [cost],
                    'Avg_Order_Value': [avg_order_value]
                })
                
                # Make prediction
                predicted_revenue = model.predict(input_data)[0]
                
                # Calculate metrics
                estimated_conversions = predicted_revenue / avg_order_value if avg_order_value > 0 else 0
                roi = predicted_revenue / cost if cost > 0 else 0
                
                # Get ROI category and recommendation
                roi_category, roi_color, recommendation = get_roi_category_and_recommendation(roi)
                
                # Display metrics
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric(
                        label="💰 Predicted Revenue",
                        value=f"${predicted_revenue:,.2f}",
                        help="Expected revenue from this campaign"
                    )
                
                with metric_col2:
                    st.metric(
                        label="🛍️ Estimated Conversions",
                        value=f"{estimated_conversions:.0f}",
                        help="Number of expected conversions"
                    )
                
                # ROI Analysis
                st.subheader("📈 ROI Analysis")
                
                roi_col1, roi_col2 = st.columns([1, 2])
                
                with roi_col1:
                    st.metric(
                        label="ROI Ratio",
                        value=f"{roi:.2f}x",
                        help="Return on Investment ratio"
                    )
                
                with roi_col2:
                    if roi_category == "High":
                        st.success(f"🟢 **{roi_category} ROI** (>{1.0:.1f})")
                    elif roi_category == "Medium":
                        st.warning(f"🟡 **{roi_category} ROI** ({0.3:.1f}-{1.0:.1f})")
                    else:
                        st.error(f"🔴 **{roi_category} ROI** (<{0.3:.1f})")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                st.info(f"**Action:** {recommendation}")
                
                # Additional insights
                profit_loss = predicted_revenue - cost
                profit_margin = (profit_loss / predicted_revenue * 100) if predicted_revenue > 0 else 0
                
                st.subheader("📋 Campaign Summary")
                summary_data = {
                    "Metric": ["Investment", "Predicted Revenue", "Profit/Loss", "Profit Margin", "Break-even Conversions"],
                    "Value": [
                        f"${cost:,.2f}",
                        f"${predicted_revenue:,.2f}",
                        f"${profit_loss:,.2f}",
                        f"{profit_margin:.1f}%",
                        f"{cost/avg_order_value:.0f}" if avg_order_value > 0 else "N/A"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # GenAI Prompt Section
                st.subheader("🤖 Generative AI Prompt")
                st.markdown("Use this prompt with your favorite AI assistant for deeper campaign analysis:")
                
                campaign_data = {
                    'campaign_name': campaign_name,
                    'campaign_type': campaign_type,
                    'audience': audience,
                    'cost': cost,
                    'predicted_revenue': predicted_revenue,
                    'roi': roi
                }
                
                genai_prompt = create_genai_prompt(campaign_data)
                
                st.code(genai_prompt, language="text")
                
                # Copy button functionality
                if st.button("📋 Copy Prompt to Clipboard", help="Click to copy the prompt"):
                    st.toast("Prompt copied to clipboard! 📋")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check that your model file is compatible and contains the expected features.")
        
        else:
            st.info("👆 Fill out the form and click 'Predict Revenue & ROI' to see results.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>Marketing Campaign ROI Predictor | Powered by Streamlit & scikit-learn</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
