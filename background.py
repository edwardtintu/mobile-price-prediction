import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Background Image Test",
    page_icon="🌄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Function to add full-screen background image using custom CSS
def add_background_image():
    """
    Adds a background image to the Streamlit app using custom CSS.
    """
    # Corrected file path
    image_path = r"C:\Users\EDWARD\Desktop\MobilePricePrediction-main\background.jpg.jpeg"



    # Add background image with full-screen style
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("file:///{image_path}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            height: 100vh;
            width: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add the background image
add_background_image()

# Sample content
st.title("Test Background Image")
st.write("This is a test app to demonstrate adding a background image.")
