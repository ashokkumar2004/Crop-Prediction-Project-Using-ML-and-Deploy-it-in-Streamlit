import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open('crop_prediction_model svc.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the input columns
input_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
inp_range = [(0, 140),
 (5, 145),
 (5, 205),
 (8.825674745, 43.67549305),
 (14.25803981, 99.98187601),
 (3.504752314, 9.93509073),
 (20.21126747, 298.5601175)]
# Streamlit application title and description
st.title("Predictive Model Application")
st.write("This application uses a pre-trained model to make predictions based on 7 numerical input features.")

# Sidebar inputs
st.sidebar.header("Input Features")
inputs = {}
for  i in range(7):
    inputs[input_columns[i]] = st.sidebar.slider(f"Enter value for {input_columns[i]}",*inp_range[i],inp_range[i][0])

# Prepare the input data for prediction
input_data = pd.DataFrame([inputs])

# Display the input data
st.write("Input Data:")
st.dataframe(input_data)
labels = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 
          'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 
          'coconut', 'cotton', 'jute', 'coffee']
image_urls = [
    "https://assets.thehansindia.com/h-upload/feeds/2019/07/13/195638-paddy-crop.jpg",
    "https://seed2plant.in/cdn/shop/products/maizeseeds_800x.jpg?v=1604034397",
    "https://i0.wp.com/post.medicalnewstoday.com/wp-content/uploads/sites/3/2022/04/chickpeas_closeup_1296x728_header-1024x575.jpg?w=1155&h=1528",
    "https://i0.wp.com/images-prod.healthline.com/hlcmsresource/images/AN_images/kidney-beans-1296x728-feature.jpg?w=1155&h=1528",
    "https://thumbs.dreamstime.com/b/lot-pigeon-pea-background-uses-30186612.jpg",
    "https://tiimg.tistatic.com/fp/1/006/986/dried-moth-bean-817.jpg",
    "https://www.wellandgood.com/wp-content/uploads/2014/11/Stocksy-Mung-Bean-Julie-Rideout.jpg",
    "https://4.imimg.com/data4/HG/FS/MY-23833905/black-gram-seeds-500x500.jpeg",
    "https://www.wikihow.com/images/thumb/e/ec/Make-Lentils-Step-1-Version-2.jpg/v4-460px-Make-Lentils-Step-1-Version-2.jpg",
    "https://images.healthshots.com/healthshots/en/uploads/2021/09/27184641/pomegranate-1600x900.jpg",
    "https://www.daysoftheyear.com/cdn-cgi/image/dpr=1%2Cf=auto%2Cfit=cover%2Cheight=650%2Cq=70%2Csharpen=1%2Cwidth=956/wp-content/uploads/banana-day1-scaled.jpg",
    "https://www.netmeds.com/images/cms/wysiwyg/blog/2019/04/Raw_Mango_898.jpg",
    "https://assets.epicurious.com/photos/55e4c39fcf90d6663f727a74/16:9/w_4592,h_2583,c_limit/shutterstock_209917372.jpg",
    "https://www.syngenta.co.in/sites/g/files/kgtney376/files/styles/syngenta_large_4_3/public/media/image/2022/05/27/watermelon-landing-page-banner.jpg?itok=H3wLGfaj",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRrutPoPYxvVNXh1R6p4l8EN3mH9IpMNHDarw&usqp=CAU",
    "https://cdn.britannica.com/22/187222-050-07B17FB6/apples-on-a-tree-branch.jpg",
    "https://cdn.britannica.com/24/174524-050-A851D3F2/Oranges.jpg",
    "https://t4.ftcdn.net/jpg/05/41/44/55/360_F_541445577_1i2kIGN20SH2Jy9gkkuIfPO2yWsOXNEQ.jpg",
    "https://www.foodnavigator.com/var/wrbm_gb_food_pharma/storage/images/_aliases/wrbm_large/media/images/coconut-cherrybeans/16954671-1-eng-GB/coconut-cherrybeans.jpg",
    "https://cdn.shopify.com/s/files/1/2694/3724/articles/Blog_Title_3_ba3adee4-747e-4de0-81a1-d0f08931f119_600x.jpg?v=1630947563",
    "https://thumbs.dreamstime.com/z/jute-plant-field-cultivation-assam-india-green-tall-plants-leaves-agricultural-crops-agriculture-asia-asian-background-154951730.jpg",
    "https://i0.wp.com/post.medicalnewstoday.com/wp-content/uploads/sites/3/2022/04/can_coffee_cause_cancer_1296x728_header-1024x575.jpg?w=1155&h=1528"
]


d =dict(zip(labels,image_urls))
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    image_url = d[prediction]
    st.write(f"Prediction: {prediction}")
    st.image(image_url)
