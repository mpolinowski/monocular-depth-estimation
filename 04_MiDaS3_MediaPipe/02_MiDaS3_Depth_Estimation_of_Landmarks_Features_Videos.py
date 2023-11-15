import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from scipy.interpolate import RectBivariateSpline
import torch

test_videos = glob('assets/*.mp4')


#Initializing the body landmarks detection module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)


# downloading the Midas model from TorchHub.
# model_type = "DPT_Large"     # MiDaS v3 - Large (1.28GM) (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid (470M) (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small (82M) (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)

# use GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)


# Use transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

alpha = 0.2
previous_depth = 0.0
depth_scale = 1.0


#Applying exponential moving average filter
def dist_avg_filter(current_depth):
    global previous_depth
    avg_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = avg_depth  # Update the previous depth value
    return avg_depth


#Define depth to distance
def depth_to_distance(depth_value,depth_scale):
    return 1.0 / (depth_value*depth_scale)

def depth_to_distance1(depth_value,depth_scale):
    return -1.0 / (depth_value*depth_scale)


cap = cv.VideoCapture(test_videos[1])
while cap.isOpened():
    ret, frame = cap.read()

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect the body landmarks in the frame
    results = pose.process(img)

    # Check if landmarks are detected
    if results.pose_landmarks is not None:
        # Draw Landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

        # Extract Landmark Coordinates
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

        waist_landmarks = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]]

        mid_point = ((waist_landmarks[0].x + waist_landmarks[1].x) / 2, (waist_landmarks[0].y + waist_landmarks[1].y) / 2,(waist_landmarks[0].z + waist_landmarks[1].z) /2)
        # print('INFO ::::::: ', mid_point)
        mid_x,mid_y,_ = mid_point

        
        imgbatch = transform(img).to(device)

        # Making a prediction
        with torch.no_grad():
            prediction = midas(imgbatch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()

        output = prediction.cpu().numpy()
        output_norm = cv.normalize(output, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        # Creating a spline array of non-integer grid
        h, w = output_norm.shape
        x_grid = np.arange(w)
        y_grid = np.arange(h)

        # Create a spline object using the output_norm array
        spline = RectBivariateSpline(y_grid, x_grid, output_norm)
        depth_mid_filt = spline(mid_y,mid_x)
        depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
        depth_mid_filt = (dist_avg_filter(depth_midas)/10)[0][0]

        # write distance to image
        cv.putText(
        img, "Dis: " + str(np.format_float_positional(depth_mid_filt, precision=2)),
        (20,50),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        1
        )
        cv.imshow('Prediction',img)

    if cv.waitKey(1) &0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()


