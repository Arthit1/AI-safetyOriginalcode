import streamlit as st
import torch
from PIL import Image
import os
from datetime import datetime
from detect import detect  # Import detect function from your custom module
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)

# Function to process image input
def imageInput(device, src):
    # Global variables to count helmet and no-helmet detections
    global counter1, counter2
    
    if src == 'อัปโหลดรูปภาพ':
        # File uploader
        image_file = st.file_uploader("ตรวจสอบรูปภาพ", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        
        if image_file is not None:
            # Open and display the uploaded image
            img = Image.open(image_file)
            max_width = 800
            max_height = 600
            img.thumbnail((max_width, max_height))
            with col1:
                st.image(img, caption='รูปภาพที่นำเข้ามา', use_column_width='always')
                
            # Save the uploaded image
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # Call detection function
            pred = detect(model, imgpath, device=device)
            pred.render()  # render bbox in image
            
            # Count helmet and no-helmet detections
            objects_Helm = ['Helm']
            objects_NoHelm = ['NoHelm']
            num_Helm = sum(1 for obj in pred.names[0] if obj in objects_Helm)
            num_NoHelm = sum(1 for obj in pred.names[0] if obj in objects_NoHelm)
            counter1 += num_Helm  # Increment Helm counter
            counter2 += num_NoHelm  # Increment NoHelm counter

            # Display prediction
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='ผลลัพธ์จากการตรวจสอบ', use_column_width='always')

# Function to process video input
def videoInput(device, src):
    # Global variables to count helmet and no-helmet detections
    global counter1, counter2
    
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    
    if uploaded_video is not None:
        ts = datetime.timestamp(datetime.now())
        pp = ts
        imgpath = os.path.join('data/uploads', str(ts)+uploaded_video.name)
        outputpath = os.path.join('data/video_output', os.path.basename(imgpath))

        with open(imgpath, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(imgpath, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("วีดีโอที่ถูกนำเข้ามา")
        
        # Call detection function
        detect(model, imgpath, device=device, project=outputpath)

        # Provide download link for the processed video
        st_video2 = open(outputpath+"/exp/"+ str(pp)+uploaded_video.name, 'rb')
        video_bytes2 = st_video2.read()
        st.download_button(label="Download video file", data=video_bytes2,file_name='video_clip.mp4')
        st.write("ผลลัพท์การตรวจสอบ")

# Main function
def main():
    # Global variables to count helmet and no-helmet detections
    global counter1, counter2
    
    # Initialize counters
    counter1 = 0
    counter2 = 0

    # Sidebar
    st.sidebar.title('🧠 Ai Ensure Worker Safety')
    datasrc = st.sidebar.radio("เลือกประเภทรูปแบบการนำเข้า", ['อัปโหลดรูปภาพ'])

    # Options
    option = st.sidebar.radio("ระบุประเภทข้อมูล", ['Image', 'Video'], disabled=False)
    
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("ประมวลผลโดยใช้", ['cpu', 'cuda'], disabled=False, index=1)
    else:
        deviceoption = st.sidebar.radio("ประมวลผลโดยใช้", ['cpu', 'cuda'], disabled=True, index=0)

    # Header
    st.header('👷 Ai Ensure Worker Safety Demo')
    
    # Image option
    if option == "Image":    
        imageInput(deviceoption, datasrc)
        valuesimg = st.selectbox('Example', ('Case 1', 'Case 2', 'Case 3'))
        
        if valuesimg == "Case 1":
            st.image(image1, caption='picture 1')
            st.write("ผลลัพท์การตรวจสอบ")
        elif valuesimg == "Case 2":
            st.image(image2, caption='picture 2')
            st.write("ผลลัพท์การตรวจสอบ")
        elif valuesimg == "Case 3":
            st.image(image3, caption='picture 3')
            st.write("ผลลัพท์การตรวจสอบ")

    # Video option
    elif option == "Video": 
        videoInput(deviceoption, datasrc)
        values = st.selectbox('Example', ('Case 1', 'Case 2', 'Case 3'))
        
        if values == "Case 1":
            st_video_test1 = open("data/outputs/test.mp4", 'rb')
            st.video(st_video_test1)
            st.write("ผลลัพท์การตรวจสอบ")
        elif values == "Case 2":
            st_video_test2 = open("data/outputs/test1.mp4", 'rb')
            st.video(st_video_test2)
            st.write("ผลลัพท์การตรวจสอบ")
        elif values == "Case 3":
            st_video_test3 = open("data/outputs/test4.mp4", 'rb')
            st.video(st_video_test3)
            st.write("ผลลัพท์การตรวจสอบ")

    # Sidebar text
    st.sidebar.text(f"จำนวนคนที่สวมใส่หมวก:{counter1}")
    st.sidebar.text(f"จำนวนคนที่ไม่สวมใส่หมวก:{counter2}")

if __name__ == '__main__':
    main()

@st.cache
def loadModel():
    start_dl = time.time()
    model_file = "models/best.pt" 
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
loadModel()
