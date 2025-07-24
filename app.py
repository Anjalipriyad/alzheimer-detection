import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import io
from keras.models import load_model

# --- UI Enhancements: Set page configuration ---
st.set_page_config(
    page_title="Alzheimer's Detection",
    page_icon="🧠", # Brain emoji as icon
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="expanded"
)

# Load the Keras .h5 model
try:
    # Use the absolute path provided by the user. If you deploy this,
    # consider changing to a relative path like "model/your_model.h5"
    model_path = "model/my_model.h5" # Changed to .h5
    model = load_model(model_path) # Load the Keras model
    st.sidebar.success(f"Model loaded successfully! ✅") # Success message in sidebar
except Exception as e:
    st.error(f"Error loading Keras .h5 model: {e}. Please ensure '{model_path}' is accessible "
             f"and it's a valid Keras .h5 model file.")
    st.stop() # Stop execution if the model can't be loaded

# Dynamically get IMG_HEIGHT and IMG_WIDTH from the model's input shape
# Assuming input shape is [None, height, width, channels] for Keras models
# Your notebook confirms 128x128 is used for training.
IMG_HEIGHT = model.input_shape[1]
IMG_WIDTH = model.input_shape[2]

# --- FIX: Explicitly set INPUT_DTYPE to np.float32 ---
# As confirmed by your model's Rescaling layer, the model expects float32.
# We ensure the image is converted to float32 before passing to the model.
INPUT_DTYPE = np.float32 
# --- END FIX ---

# IMPORTANT: This order MUST exactly match the order of class labels
# used during your model's training. Your notebook confirms this order.
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Added a function to display model input/output details in the sidebar for debugging
def display_model_debug_details():
    with st.sidebar.expander("Model Debug Details (Click to Expand)", expanded=False):
        st.subheader("Model Details")
        st.write(f"Input Tensor Shape: `{model.input_shape}`")
        st.write(f"Input Tensor Dtype (Assumed): `{INPUT_DTYPE}`") # Updated label
        st.write(f"Output Tensor Shape: `{model.output_shape}`")
        st.write(f"Expected Image Size: `{IMG_WIDTH}x{IMG_HEIGHT}` pixels")
        st.write(f"Assumed Class Names Order: `{class_names}`")

display_model_debug_details() # Call this to display details in sidebar when app starts

# Modified model_predict function to accept image content as bytes
def model_predict(image_bytes_content):
    """
    Predicts the class of an MRI image using a Keras .h5 model.

    Args:
        image_bytes_content (bytes): The raw byte content of the image file.

    Returns:
        tuple: (predicted_class, confidence, matplotlib_figure) or (None, None, None) on error.
    """
    try:
        # Convert bytes content to a numpy array for OpenCV decoding
        file_bytes_array = np.frombuffer(image_bytes_content, np.uint8)
        st.sidebar.write(f"Debug: File bytes array shape: {file_bytes_array.shape}, dtype: {file_bytes_array.dtype}")
        
        # Decode the image using OpenCV
        img = cv2.imdecode(file_bytes_array, cv2.IMREAD_COLOR)

        # Check if cv2.imdecode successfully read the image
        if img is None:
            st.error("OpenCV could not decode the image. "
                     "It might be corrupted, empty, or in an unsupported format.")
            return None, None, None
        
        st.sidebar.write(f"Debug: Decoded image shape (BGR): {img.shape}, dtype: {img.dtype}")

        # Process the image: Convert BGR to RGB (as per notebook's predict_custom_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        st.sidebar.write(f"Debug: Image shape (RGB after cvtColor): {img.shape}, dtype: {img.dtype}")

        # Resize to model input size (128x128 as per notebook's image_size and IMG_HEIGHT/WIDTH)
        # Use INTER_AREA for downsampling (recommended for image reduction).
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        st.sidebar.write(f"Debug: Resized image shape: {img.shape}, dtype: {img.dtype}")
        
        # --- FIX: Only convert to float32, DO NOT divide by 255.0 ---
        # The model itself has a tf.keras.layers.Rescaling(1./255) layer
        # so the input to model.predict should be raw pixel values (0-255) as float32.
        img_array = img.astype(INPUT_DTYPE) 
        # --- END FIX ---

        # Add batch dimension (model expects shape like [1, H, W, C])
        img_array = np.expand_dims(img_array, axis=0)
        
        st.sidebar.write(f"Debug: Final img_array shape: {img_array.shape}, dtype: {img_array.dtype}")
        # Pixel values should now be 0-255
        st.sidebar.write(f"Debug: Final img_array min/max values (before model's rescaling): {img_array.min()}/{img_array.max()}")

        # Perform prediction using the Keras model
        prediction_raw = model.predict(img_array) # Changed for Keras model
        st.sidebar.write(f"Debug: Raw prediction output (probabilities): {prediction_raw}")
        
        predicted_class_index = np.argmax(prediction_raw)
        predicted_class = class_names[predicted_class_index]
        confidence = np.max(prediction_raw)

        # Create a matplotlib figure for visualization
        fig, ax = plt.subplots(figsize=(6, 6)) # Make plot a bit larger
        # Display the image as uint8 for correct visualization (matplotlib expects 0-255 for uint8)
        ax.imshow(img.astype("uint8")) 
        ax.axis('off')
        ax.set_title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2%}", fontsize=14, fontweight='bold')
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        return predicted_class, confidence, fig

    except Exception as e:
        st.error(f"Error during image prediction: {str(e)}")
        # Display full traceback for more detailed debugging in the Streamlit app
        st.exception(e) 
        return None, None, None

# Streamlit UI
st.title("🧠 Alzheimer's Detection from Brain MRI")
st.markdown("""
Welcome to the Alzheimer's Detection app. Upload an MRI brain scan image (JPG, JPEG, PNG) 
below to get an instant prediction on the **potential stage of Alzheimer's disease onset.**
""")

st.write("---") # Separator for visual appeal

uploaded_file = st.file_uploader("Upload an MRI image here...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    # Use a container for the image and classification results
    with st.container(border=True):
        st.subheader("Uploaded Image")
        try:
            # Read the file content ONCE
            file_content = uploaded_file.read()

            # Display the image using PIL (Image.open can read from BytesIO)
            image_display = Image.open(io.BytesIO(file_content))
            st.image(image_display, caption='MRI Image', use_column_width=True, channels="RGB")

            st.markdown("---") # Separator
            st.subheader("Classification Results")
            st.write("Processing your image, please wait...")
            
            # Show a spinner while classifying
            with st.spinner("Classifying MRI..."):
                predicted_class, confidence, fig = model_predict(file_content)

            if predicted_class is not None:
                if predicted_class == 'NonDemented':
                    st.success(f"**Prediction: {predicted_class} ✅**")
                    st.write("The model predicts no signs of Alzheimer's disease.")
                elif predicted_class == 'VeryMildDemented':
                    st.warning(f"**Prediction: {predicted_class} ⚠️**")
                    st.write("The model detects **very mild** cognitive impairment, indicating initial signs or a very early stage. Further medical consultation is recommended.")
                elif predicted_class == 'MildDemented':
                    st.error(f"**Prediction: {predicted_class} 🚨**")
                    st.write("The model detects **mild** Alzheimer's disease. This suggests noticeable cognitive decline. Please consult a medical professional.")
                else: # ModerateDemented
                    st.error(f"**Prediction: {predicted_class} 🚨**")
                    st.write("The model detects **moderate** Alzheimer's disease. This indicates more significant cognitive and functional impairment. Immediate medical attention is advised.")

                st.info(f"Confidence: **{confidence:.2%}**")
                
                # Display the matplotlib figure within the container
                st.pyplot(fig) 
                plt.close(fig) # IMPORTANT: Close the figure to free up memory and prevent warnings
            else:
                st.error("Could not classify the image. Please try again.")

        except Exception as e:
            st.error(f"An unexpected error occurred while loading or processing the image: {str(e)}")
            st.warning("Please try uploading a different image or ensure it's a valid image file (JPG, JPEG, PNG).")

st.write("---") # Footer separator
st.markdown("Disclaimer: This tool is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns.")
