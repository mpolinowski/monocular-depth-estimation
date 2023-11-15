import cv2 as cv
from glob import glob
import math
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

# instantiate pose landmark predictor
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# reduce twitchiness by averaging detected depth
alpha = 0.6
previous_depth = 0.0

def dist_avg_filter(current_depth):
    global previous_depth
    avg_depth = alpha * current_depth + (1 - alpha) * previous_depth # exponential moving average
    previous_depth = avg_depth  # update the previous depth value
    return avg_depth

# create normalized depth scale
def depth_to_distance(depth_value, depth_scale):
  return - 1.0 / (depth_value * depth_scale)

test_videos = glob('assets/*.mp4')
cap = cv.VideoCapture(test_videos[0])

while cap.isOpened():
  # start extracting frames
  ret, frame = cap.read()
  # convert to RGB
  img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
  # get pose landmarks
  results = pose.process(img)

  # check to see if body landmarks are being detected  
  if results.pose_landmarks is not None:
    landmarks = []

    # get nose z-position for depth estimation
    for landmark in results.pose_landmarks.landmark:
      landmarks.append((landmark.x, landmark.y, landmark.z))

    nose_landmark = landmarks[mp_pose.PoseLandmark.NOSE.value]
    _, _, nose_z = nose_landmark

    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(img,results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # convert the distance to your own requirement e.g. 0 to 1
    filter = dist_avg_filter(nose_z)
    distance = depth_to_distance(filter,1)

    # write distance to image
    cv.putText(
      img, "Norm distance: " + str(np.format_float_positional(distance, precision=2)),
      (20,50),
      cv.FONT_HERSHEY_SIMPLEX,
      1,
      (255,255,255),
      3
    )
    cv.imshow('Prediction',img)

  if cv.waitKey(1) & 0xFF == ord('q'):
    cap.release()
    cv.destroyAllWindows() 