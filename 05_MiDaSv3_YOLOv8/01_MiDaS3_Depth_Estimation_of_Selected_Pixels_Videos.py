import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import center_of_mass
import torch

test_videos = glob('assets/*.mp4')

# downloading the Midas model from TorchHub.
# model_type = "DPT_Large"     # MiDaS v3 - Large (1.28GM) (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid (470M) (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small (82M) (lowest accuracy, highest inference speed)

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


# define depth to distance
def depth_to_distance(depth_value,depth_scale):
    return 1.0 / (depth_value*depth_scale)


cap = cv.VideoCapture(test_videos[0])
while cap.isOpened():
    ret, frame = cap.read()

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
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

    # normalize depth map
    output = prediction.cpu().numpy()
    output_norm = cv.normalize(output, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    # get center cords of image
    center_y, center_x = center_of_mass(output_norm)
    # Creating a spline array of non-integer grid
    h, w = output_norm.shape
    x_grid = np.arange(w)
    y_grid = np.arange(h)

    # Create a spline object using the output_norm array
    spline = RectBivariateSpline(y_grid, x_grid, output_norm)
    depth_mid_filt = spline(center_x,center_y)
    depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
    depth_mid_filt = (dist_avg_filter(depth_midas)/10)[0][0]

    # write distance to image
    cv.putText(
        img, "Dis: " + str(np.format_float_positional(depth_mid_filt, precision=2)),
        (20,60), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1
    )
    # mark center
    cv.circle(img,(int(center_x), int(center_y)), 40, (255,255,255), 20)
    cv.imshow('Prediction',img)

    if cv.waitKey(1) &0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()


