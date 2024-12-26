import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load model and scaler
def load_model_and_scaler():
    model = joblib.load('mobile_price_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

# Map price range to descriptions
PRICE_RANGE_DESCRIPTIONS = {
    0: "Low cost",
    1: "Medium cost",
    2: "High cost",
    3: "Very high cost"
}

# Get user input from the Streamlit interface
def get_user_input(feature_columns, feature_descriptions):
    """
    Get user input for mobile specifications with descriptions and ranges
    """
    user_data = {}
    for feature, label in feature_columns.items():
        user_data[feature] = st.text_input(
            f"{label}",
            placeholder=f"Enter {label} value",
            help=f"{feature_descriptions[feature]}"
        )
    user_data = {k: float(v) if v else 0.0 for k, v in user_data.items()}
    return pd.DataFrame([user_data])

# Streamlit App
def main():
    st.set_page_config(
        page_title="Mobile Price Prediction App",
        page_icon="📱",
        layout="wide"
    )

    # App Header
    st.markdown(
        """
        <div style="background-color:#4CAF50; padding:10px; border-radius:10px;">
            <h1 style="text-align: center; color: white;">Mobile Price Prediction</h1>
            <p style="text-align: center; color: #f2f2f2;">Predict the price range of a mobile phone based on its specifications.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Information
    st.sidebar.markdown(
        """
        <div style="background-color:#FFC107; padding:10px; border-radius:10px;">
            <h2 style="color: white; text-align: center;">Instructions</h2>
            <p style="color: black; text-align: justify;">
            Enter the specifications of your mobile phone in the input fields provided below. After submitting, 
            you will receive a prediction of the mobile phone's price range along with an explanation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load the model and scaler
    model, scaler = load_model_and_scaler()

    # Feature columns with full forms
    feature_columns = {
        'battery_power': "Battery Power (mAh)",
        'blue': "Bluetooth Availability (0: No, 1: Yes)",
        'clock_speed': "Clock Speed (GHz)",
        'dual_sim': "Dual SIM (0: No, 1: Yes)",
        'fc': "Front Camera Resolution (MP)",
        'four_g': "4G Support (0: No, 1: Yes)",
        'int_memory': "Internal Memory (GB)",
        'm_dep': "Mobile Depth (cm)",
        'mobile_wt': "Mobile Weight (grams)",
        'n_cores': "Number of Processor Cores",
        'pc': "Primary Camera Resolution (MP)",
        'px_height': "Pixel Resolution Height",
        'px_width': "Pixel Resolution Width",
        'ram': "RAM (MB)",
        'sc_h': "Screen Height (cm)",
        'sc_w': "Screen Width (cm)",
        'talk_time': "Battery Talk Time (hours)",
        'three_g': "3G Support (0: No, 1: Yes)",
        'touch_screen': "Touch Screen (0: No, 1: Yes)",
        'wifi': "WiFi Support (0: No, 1: Yes)"
    }

    # Feature descriptions with valid ranges
    feature_descriptions = {
        'battery_power': "Battery capacity in mAh (Range: 500 - 5000)",
        'blue': "Bluetooth availability (0: No, 1: Yes)",
        'clock_speed': "Processor speed in GHz (Range: 0.5 - 3.0)",
        'dual_sim': "Dual SIM support (0: No, 1: Yes)",
        'fc': "Front camera resolution in MP (Range: 0 - 20)",
        'four_g': "4G availability (0: No, 1: Yes)",
        'int_memory': "Internal memory in GB (Range: 2 - 256)",
        'm_dep': "Mobile depth in cm (Range: 0.1 - 2.0)",
        'mobile_wt': "Mobile weight in grams (Range: 80 - 300)",
        'n_cores': "Number of processor cores (Range: 1 - 8)",
        'pc': "Primary camera resolution in MP (Range: 0 - 20)",
        'px_height': "Pixel resolution height (Range: 0 - 2000)",
        'px_width': "Pixel resolution width (Range: 0 - 2000)",
        'ram': "RAM in MB (Range: 256 - 8192)",
        'sc_h': "Screen height in cm (Range: 5 - 20)",
        'sc_w': "Screen width in cm (Range: 2 - 10)",
        'talk_time': "Battery talk time in hours (Range: 2 - 24)",
        'three_g': "3G availability (0: No, 1: Yes)",
        'touch_screen': "Touchscreen availability (0: No, 1: Yes)",
        'wifi': "WiFi availability (0: No, 1: Yes)"
    }

    # User input section
    st.markdown("### Mobile Specifications")
    user_input_df = get_user_input(feature_columns, feature_descriptions)

    # Display the user input table
    if st.button("Show Input Summary"):
        st.markdown("#### Your Input Data")
        st.dataframe(user_input_df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]}]
        ))

    # Scale the input data
    scaled_input = scaler.transform(user_input_df)

    # Prediction Section
    st.markdown("### Predict Mobile Price Range")
    if st.button("Predict Now"):
        prediction = model.predict(scaled_input)
        price_range = prediction[0]

        # Display prediction result with description
        st.markdown(
            f"""
            <div style="background-color:#FFC107; padding:20px; border-radius:10px;">
                <h2 style="text-align: center; color: black;">Predicted Price Range: {price_range}</h2>
                <p style="text-align: center; color: black;">{PRICE_RANGE_DESCRIPTIONS[price_range]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success("Prediction completed successfully!")

    # Footer Section
    st.markdown(
        """
        <hr>
        <p style="text-align: center; color: #808080;">Developed by Edward | Powered by Streamlit</p>
        """,
        unsafe_allow_html=True,
    )

# Run the app
if __name__ == "__main__":
    main()
