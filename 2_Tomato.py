import streamlit as st
#for defining the loss in compile function
import tensorflow as tf

#for loading the model trained earlier for prediction purpose
from tensorflow.keras.models import load_model

#for loading the image and making changes in it
#to make it fit for prediction purpose
from tensorflow.keras.preprocessing import image

import numpy as np

from warnings import filterwarnings
filterwarnings("ignore")

#for adding background image
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('tomato_background.jpeg')

#adding title
st.title('TOMATO LEAF DISEASE DETECTOR')

solutions = {
    "Tomato - Bacteria Spot Disease": """
    **Immediate Action:** Remove and destroy affected leaves to reduce spread.\n
    **Copper Sprays:** Apply copper-based bactericides weekly to manage the disease.\n
    **Preventive Measures:** Avoid overhead watering to reduce leaf wetness.\n
    **Sanitation:** Disinfect gardening tools and avoid working with wet plants.
    """,
    "Tomato - Early Blight Disease": """
    **Immediate Action:** Remove and destroy infected plant parts.\n
    **Fungicide Application:** Apply fungicides like chlorothalonil or copper-based sprays at the first sign of disease.\n
    **Crop Rotation:** Rotate crops and avoid planting tomatoes in the same soil for at least two years.\n
    **Healthy Practices:** Ensure proper spacing and air circulation around plants.
    """,
    "Tomato - Healthy and Fresh": """
    **Action:** Maintain regular crop monitoring and good agricultural practices.\n
    **Preventive Measures:** Ensure proper irrigation, use balanced fertilizers, and control pests.\n
    **Soil Health:** Rotate crops to maintain soil fertility and prevent disease build-up.\n
    **Surveillance:** Continue regular inspections to catch any early signs of disease or pests.
    """,
    "Tomato - Late Blight Disease": """
    **Immediate Action:** Remove and destroy all infected plants and plant debris.\n
    **Fungicide Treatment:** Use fungicides such as metalaxyl or copper-based sprays regularly during humid conditions.\n
    **Resistant Varieties:** Plant tomato varieties that are resistant to late blight.\n
    **Water Management:** Avoid overhead irrigation and ensure proper drainage to reduce leaf wetness duration.
    """,
    "Tomato - Leaf Mold Disease": """
    **Immediate Action:** Remove and destroy affected leaves to reduce inoculum.\n
    **Ventilation:** Improve air circulation by pruning and spacing plants appropriately.\n
    **Fungicide Application:** Use fungicides like chlorothalonil or copper-based sprays as preventive measures.\n
    **Humidity Control:** Reduce humidity in greenhouses by venting and reducing plant density.
    """,
    "Tomato - Septoria Leaf Spot Disease": """
    **Immediate Action:** Remove and destroy infected leaves to prevent spread.\n
    **Fungicide Application:** Apply fungicides like chlorothalonil or mancozeb at the first sign of disease.\n
    **Watering Practices:** Water at the base of plants to avoid wetting the foliage.\n
    **Sanitation:** Rotate crops and avoid planting tomatoes in the same soil for at least two years.
    """,
    "Tomato - Target Spot Disease": """
    **Immediate Action:** Remove and destroy affected plant parts.\n
    **Fungicide Treatment:** Use fungicides like azoxystrobin or copper-based sprays regularly.\n
    **Crop Rotation:** Rotate crops to prevent the build-up of soil-borne pathogens.\n
    **Healthy Practices:** Maintain good air circulation and avoid overhead irrigation.
    """,
    "Tomato - Tomato Yellow Leaf Curl Virus Disease": """
    **Immediate Action:** Remove and destroy infected plants immediately to prevent spread.\n
    **Insect Control:** Use insecticides or biological controls to manage whitefly populations, which spread the virus.\n
    **Resistant Varieties:** Plant tomato varieties that are resistant to TYLCV.\n
    **Preventive Measures:** Use reflective mulches to repel whiteflies and install physical barriers.
    """,
    "Tomato - Two Spotted Spider Mite Disease": """
    **Immediate Action:** Wash off mites with a strong stream of water.\n
    **Biological Control:** Introduce natural predators like ladybugs or predatory mites.\n
    **Miticide Application:** Use miticides if the infestation is severe and other methods are not effective.\n
    **Healthy Practices:** Maintain proper watering and avoid plant stress to reduce susceptibility to mites.
    """
}

# target dimensions of image
img_width, img_height = 128, 128

#for uploading the image
test= st.file_uploader("",type= ["png","jpg","jpeg"])

if test!=None and st.button('Predict'):
    # load the model we saved
    model = load_model('tomatoModel.h5')

    # predicting images
    #creating an object of image and changing its size to required size
    img = image.load_img(test, target_size=(img_width, img_height))

    #converting the image to array
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    image = np.vstack([x])
    
    #gives probability for each class
    prob = model.predict(image)

    #possible classes
    class_= ['Tomato - Bacteria Spot Disease', 'Tomato - Early Blight Disease', 'Tomato - Healthy and Fresh',
             'Tomato - Late Blight Disease','Tomato - Leaf Mold Disease','Tomato - Septoria Leaf Spot Disease',
             'Tomato - Target Spot Disease','Tomato - Tomoato Yellow Leaf Curl Virus Disease',
             'Tomato - Two Spotted Spider Mite Disease']

    #print class with maximum probability
    st.metric(label= "Predicted Class", value= class_[np.argmax(prob)])

    st.write("### Recommended Actions and Information:")
    st.write(solutions[class_[np.argmax(prob)]])