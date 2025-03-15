
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Run this code to save all uploaded files from the sample_data directory to the current directory

import os
import shutil

# Set the source directory where your uploaded files are located
upload_dir = '/content/sample_data'  # Update this path to point to your files

# Get the current directory where your Jupyter Notebook is located
current_dir = os.getcwd()

# Iterate through files in the upload directory and copy them to the current directory
for file in os.listdir(upload_dir):
    if os.path.isfile(os.path.join(upload_dir, file)):
        # Copy the file and display a message
        source_file = os.path.join(upload_dir, file)
        destination_file = os.path.join(current_dir, file)
        shutil.copy(source_file, destination_file)
        print(f"Saved: {file}")

# List all the files that have been copied to the current directory
print("Files saved in the current directory:", os.listdir(current_dir))


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #RUN THS SNIPPET FIRST DOWNLOAD MODEL FOR FACE LANDMARKS FROM GITHUB
!pip install dlib opencv-python-headless imutils matplotl

import dlib
import urllib.request
import bz2
import os

# Define the URL of the compressed model file on GitHub
model_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"

# Define the local file path where you want to save the model
model_path = "shape_predictor_68_face_landmarks.dat"

# Check if the model file already exists, if not, download and extract it
if not os.path.exists(model_path):
    print("Downloading and extracting the model...")
    urllib.request.urlretrieve(model_url, model_path + ".bz2")
    with bz2.BZ2File(model_path + ".bz2", "rb") as source, open(model_path, "wb") as dest:
        dest.write(source.read())
    print("Model downloaded and extracted successfully!")

# Load the pre-trained shape predictor model for facial landmarks
predictor = dlib.shape_predictor(model_path)

#------------------------------------------------------------

#RUN THIS SNIPPET NEXT TO GET DLIB AND OTHER LIBRARIES
# Install dlib without uninstalling it and without CUDA support
#!pip install dlib --no-cache-dir --force-reinstall
import os

# Force dlib to use CPU (hide GPUs)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import other necessary libraries
import dlib
# Import other libraries and modules you need for your project

import urllib.request
import bz2
import os

# Replace the model_path variable with the specific path to the model file
model_path = "/content/dlib_face_recognition_resnet_model_v1.dat"

# Check if the model file already exists, if not, download and extract it
if not os.path.exists(model_path):
    print("Downloading and extracting the model...")
    urllib.request.urlretrieve(model_url, model_path + ".bz2")
    with bz2.BZ2File(model_path + ".bz2", "rb") as source, open(model_path, "wb") as dest:
        dest.write(source.read())
    print("Model downloaded and extracted successfully!")

# Load the pre-trained face recognition model
facerec = '/content/dlib_face_recognition_resnet_model_v1.dat'

#------------------------------------------------------------------------

#RUN THIS PREDICTOR MODEL FOR FACIAL LANDMARKS

!pip install --upgrade pip
# First, install the necessary libraries
!pip install dlib opencv-python-headless imutils moviepy
!pip install opencv-python-headless
!pip install moviepy
!pip install ffmpeg-python

import cv2
import dlib
import numpy as np
import moviepy.editor as mp
import time

from IPython.display import Audio, display, Image
from moviepy.editor import AudioFileClip
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageClip
from moviepy.editor import CompositeVideoClip
from moviepy.editor import VideoFileClip, CompositeVideoClip, clips_array
from moviepy.editor import AudioFileClip

#----------------------------------------------------------------

# Load pre-trained shape predictor model for facial landmarks
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor_path)

# Load pre-trained deep learning model for face detection
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()

# Load the song
song_url = "/content/sample_data/when_you_believe_audio.mp3"
song_clip = mp.AudioFileClip(song_url)
song_duration = 110  # 1:50 (110 seconds)

# Load the input JPEG images
input_image_path1 = '/content/sample_data/IMG-Singer1.jpg'
image1 = cv2.imread(input_image_path1)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Load the input JPEG images
input_image_path2 = '/content/sample_data/IMG-Singer2.jpg'
image2 = cv2.imread(input_image_path2)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detect faces in the input images
faces1 = detector(gray1)
faces2 = detector(gray2)

# Assuming there's only one face detected in each image
if len(faces1) > 0 and len(faces2) > 0:
    face1 = faces1[0]
    face2 = faces2[0]

    # Get facial landmarks for the first image
    landmarks1 = predictor(gray1, face1)
    mouth_coords1 = [(landmarks1.part(i).x, landmarks1.part(i).y) for i in range(48, 68)]

    # Get facial landmarks for the second image
    landmarks2 = predictor(gray2, face2)
    mouth_coords2 = [(landmarks2.part(i).x, landmarks2.part(i).y) for i in range(48, 68)]

    # Define mouth region coordinates for the first image
    x1, y1, w1, h1 = cv2.boundingRect(np.array(mouth_coords1))

    # Define mouth region coordinates for the second image
    x2, y2, w2, h2 = cv2.boundingRect(np.array(mouth_coords2))

    # Draw a green square around the detected face in the first image
    cv2.rectangle(image1, (face1.left(), face1.top()), (face1.right(), face1.bottom()), (0, 255, 0), 2)

    # Draw a red square around the mouth in the first image
    cv2.rectangle(image1, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

    # Draw a green square around the detected face in the second image
    cv2.rectangle(image2, (face2.left(), face2.top()), (face2.right(), face2.bottom()), (0, 255, 0), 2)

    # Draw a red square around the mouth in the second image
    cv2.rectangle(image2, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

    # Display the first image with squares and add an audio button
    image_with_squares_path1 = '/content/sample_data/image_with_squares1.jpeg'
    cv2.imwrite(image_with_squares_path1, image1)
    display(Image(filename=image_with_squares_path1, width=400))
    display(Audio(filename='/content/sample_data/when_you_believe_audio.mp3'))

    # Display the second image with squares and add an audio button
    image_with_squares_path2 = '/content/sample_data/image_with_squares2.jpeg'
    cv2.imwrite(image_with_squares_path2, image2)
    display(Image(filename=image_with_squares_path2, width=400))
    display(Audio(filename='/content/sample_data/when_you_believe_audio.mp3'))
else:
    print("No face detected in one or both input images after trying multiple rotations.")

#-----------------------------------------------------------------

# Use OpenCV to detect and track mouth movements

# Load the .jpg images for overlay
image1 = mp.ImageClip("/content/sample_data/IMG-Singer1.jpg")
image2 = mp.ImageClip("/content/sample_data/IMG-Singer2.jpg")


# Load the mouth movement videos
video1 = VideoFileClip("/content/sample_data/IMG_1947.3gp")
#video2 = VideoFileClip("/content/sample_data/IMG_1947.3gp")

# Set the duration for overlay (1:50 - 0:33 seconds)
overlay_duration1 = song_duration - 33

# Set the duration for overlay (2:27 - 0:33 seconds)
#overlay_duration2 = song_duration - 33

# Overlay the words over the images
overlay1 = mp.CompositeVideoClip([image1.set_duration(overlay_duration1)])
#overlay2 = mp.CompositeVideoClip([image2.set_duration(overlay_duration2)])

# Create the script audio
script_audio = mp.AudioFileClip("/content/sample_data/when_you_believe_audio.mp3")

# Combine images and script audio
final_clip1 = mp.clips_array([[overlay1]])
final_clip1 = final_clip1.set_audio(script_audio)

# Add an audio button to play the song
song_audio = mp.AudioFileClip("/content/sample_data/when_you_believe_audio.mp3")
final_clip1 = final_clip1.set_audio(song_audio)

# Set the duration for the images to match the videos
image1 = image1.set_duration(video1.duration)

# Create a video clip with the images side by side
side_by_side = clips_array([[image1, image2]])

# Overlay the mouth movement videos on the images
video1 = video1.set_position("center")
#video2 = video2.set_position("center")

#final_clip = CompositeVideoClip([side_by_side, video1, video2])

# Overlay the mouth movement video on top of a black background
mouth_movement_overlay1 = video1.set_position("center").on_color(
    (1920, 1080), color=(0, 0, 0), pos="center")

# Combine the mouth movement overlay with the audio
final_clip1 = CompositeVideoClip([mouth_movement_overlay1.set_audio(song_audio)])

# Write the final video
final_clip1.write_videofile("final_output.mp4", codec="libx264", audio_codec="aac")

print('Final output video printed')
# Display the audio playback button for the song
final_clip1.ipython_display(t=0, width=800)

            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    