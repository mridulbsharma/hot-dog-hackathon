import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import io

# Set page configuration
st.set_page_config(page_title="🌭 Hot Dog Classifier", page_icon="🌭", layout="wide")

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('best_hotdog_classifier.h5')
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

def preprocess_image(image):
    img = image.resize((299, 299))  # Resize to match InceptionV3 input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.

def predict_image(model, img_array):
    try:
        prediction = model.predict(img_array)
        return prediction[0][0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def create_gauge_chart(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Hot Dog Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    return fig

def main():
    st.title("🌭 Hot Dog or Not Hot Dog Classifier")
    st.write("Upload an image to see if it's a hot dog or not!")

    model = load_model()
    if model is None:
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            with st.spinner('Classifying image...'):
                img_array = preprocess_image(image)
                raw_confidence = predict_image(model, img_array)
            
            if raw_confidence is not None:
                # Flip the confidence score
                confidence = 1 - raw_confidence

                st.write("## Results")
                if confidence > 0.5:
                    st.success(f"It's a hot dog! 🌭 (Confidence: {confidence:.2%})")
                else:
                    st.error(f"It's not a hot dog! ❌ (Confidence: {(1-confidence):.2%})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Hot Dog Confidence Gauge")
                    fig = create_gauge_chart(confidence)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("### Probability Distribution")
                    chart_data = pd.DataFrame({
                        'Category': ['Hot Dog', 'Not Hot Dog'],
                        'Probability': [float(confidence), 1 - float(confidence)]
                    })
                    st.bar_chart(chart_data.set_index('Category'))
            
        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")

    st.write("### How it works")
    st.write("This app uses a deep learning model trained on thousands of images to classify whether an uploaded image contains a hot dog or not. The model was trained using transfer learning with a pre-trained convolutional neural network.")

    st.write("### Model Details")
    st.write(f"Model architecture: {model.name}")
    st.write(f"Number of layers: {len(model.layers)}")
    st.write(f"Input shape: {model.input_shape}")

    if st.checkbox("Show model summary"):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        st.code("\n".join(model_summary), language="plaintext")

    st.sidebar.title("About")
    st.sidebar.info(
        "This app demonstrates the use of deep learning for image classification. "
        "It's a fun project inspired by the 'SeeFood' app from the TV show Silicon Valley. "
        "\n\n"
        "Created by: Mridul Sharma"
    )

    st.sidebar.title("Resources")
    st.sidebar.markdown(
        """
        - [TensorFlow](https://www.tensorflow.org/)
        - [Streamlit](https://streamlit.io/)
        - [Plotly](https://plotly.com/)
        """
    )

if __name__ == "__main__":
    main()