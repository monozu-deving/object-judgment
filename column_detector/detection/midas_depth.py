import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# 모델 로딩 (최초 1회)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def estimate_depth_map(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transforms(img).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    return prediction.cpu().numpy()

def normalize_depth_map(depth_map):
    """
    0 ~ 1 사이로 정규화된 depth map 반환
    """
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    norm = (depth_map - d_min) / (d_max - d_min + 1e-6)
    return norm
