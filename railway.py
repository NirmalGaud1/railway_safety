import streamlit as st
from ultralytics import YOLO
import cv2
import google.generativeai as genai
import tempfile
import os
import numpy as np

# Configure Gemini AI
API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"  # Replace with your Gemini API key
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Use Gemini Flash model

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8m.pt')  # Use medium-sized model for better accuracy

# Define class IDs for objects of interest
CLASS_IDS = {
    "train": 7,       # COCO class ID for train
    "car": 2,         # COCO class ID for car
    "person": 0       # COCO class ID for person
}

# Function to generate a detailed description using Gemini AI
def generate_detailed_description(detected_objects):
    prompt = (
        f"The following objects were detected at a railway crossing: {detected_objects}. "
        "Provide a detailed analysis focusing on: "
        "1. The **technical details of the objects' locations** (e.g., bounding box coordinates, size, and position relative to the tracks). "
        "2. The **possible risks** associated with the detected objects. "
        "3. **Safety measures** to mitigate the risks. "
        "Avoid unnecessary explanations or background information."
    )
    response = gemini_model.generate_content(prompt)
    return response.text

# Function to process an image or video frame
def process_frame(frame):
    # Perform object detection on the frame
    results = model.predict(frame, conf=0.3)  # Lower confidence threshold

    # Initialize lists for detected objects
    detected_objects = []

    # Process detection results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)  # Get class ID of detected object
            confidence = float(box.conf)  # Get confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

            # Get the object name (e.g., train, car, person)
            object_name = model.names[class_id]

            # Add detected object to the list
            detected_objects.append({
                "name": object_name,
                "confidence": confidence,
                "bounding_box": (x1, y1, x2, y2)
            })

            # Draw bounding box on the frame
            label = f"{object_name} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Generate detailed description of the situation
    detailed_description = None
    if detected_objects:
        # Format detected objects for the description
        detected_objects_str = ", ".join(
            [f"{obj['name']} (confidence: {obj['confidence']:.2f}) at position {obj['bounding_box']}" 
             for obj in detected_objects]
        )
        detailed_description = generate_detailed_description(detected_objects_str)

    return frame, detailed_description

# Streamlit app
def main():
    st.title("🚦 Railway Crossing Safety System")
    st.write("This app detects trains, vehicles, and pedestrians at railway crossings and provides detailed safety analysis.")

    # Upload image or video
    input_type = st.radio("Select input type:", ["Image", "Video"])

    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read the image
            file_bytes = uploaded_file.read()
            frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Process the image
            processed_frame, detailed_description = process_frame(frame)

            # Display the processed image
            st.image(processed_frame, channels="BGR", caption="Processed Image", use_column_width=True)

            # Display the detailed description
            if detailed_description:
                st.subheader("Detailed Safety Analysis")
                st.write(detailed_description)
            else:
                st.info("No objects detected.")

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name

            # Process the video
            st.write("Processing video...")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error: Could not open video.")
                return

            # Display the video with detections
            stframe = st.empty()
            detailed_description = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame
                processed_frame, current_description = process_frame(frame)

                # Update the detailed description if a new one is generated
                if current_description:
                    detailed_description = current_description

                # Display the processed frame
                stframe.image(processed_frame, channels="BGR", use_column_width=True)

            cap.release()
            os.remove(video_path)  # Clean up the temporary file

            # Display the final detailed description after the video ends
            if detailed_description:
                st.subheader("Detailed Safety Analysis")
                st.write(detailed_description)
            else:
                st.info("No objects detected.")

# Run the app
if __name__ == "__main__":
    main()
