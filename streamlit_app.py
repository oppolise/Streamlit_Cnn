import streamlit as st
import torch
from PIL import Image
from prediction import pred_class
import numpy as np

# Set title 
st.title('Knee Arthritis Classification')

# Set Header 
st.header('Please upload a picture')

# Create two columns for file uploader and image display
col1, col2 = st.columns(2)

# Load Model 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('mobilenetv3_large_100_checkpoint_fold2.', map_location=device)
model.half()

# Display image & Prediction 
with col1:  # Left column for file uploader
    uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

with col2:  # Right column for displaying the uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

# Full-width section for Prediction Results
if uploaded_image is not None:
    st.markdown("---")  # Add a horizontal line for separation
    st.header("Prediction Results")

    class_name = ['Air conditioner','Car horn','Children playing','Dog bark','Drilling','Engine idling','Gun shot','Jackhammer','Siren','Street music']

    with st.expander("Show Prediction Results"):
        if st.button('Predict'):
            # Prediction class
            probli = pred_class(model, image, class_name)

            st.write("## Prediction Result")
            # Get the index of the maximum value in probli[0]
            max_index = np.argmax(probli[0])

            # Iterate over the class_name and probli lists
            for i in range(len(class_name)):
                # Set the color to blue if it's the maximum value, otherwise use the default color
                color = "blue" if i == max_index else None
                st.write(f"## <span style='color:{color}'>{class_name[i]} : {probli[0][i]*100:.2f}%</span>", unsafe_allow_html=True)
