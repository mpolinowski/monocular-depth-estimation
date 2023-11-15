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


cap = cv.VideoCapture(test_videos[0])
while cap.isOpened():
    ret, frame = cap.read()

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect the body landmarks in the frame
    results = pose.process(img)

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

        cv.imshow('Prediction',output_norm)

    if cv.waitKey(1) &0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()


