import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from torchvision import transforms

# url2tensor
def url_to_tensor(url):
    response = requests.get(url)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # C H W
    tensor = transform(image)

    # B H W C
    return tensor.permute(1, 2, 0).unsqueeze(0)

def create_highlight_mask(img_rgb, threshold=200):
    # 创建高光区域的mask
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def create_blur_mask(img_rgb, window_size=15, var_threshold=50):
    # 创建模糊区域的mask
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = cv2.blur(laplacian ** 2, (window_size, window_size))
    norm_variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(norm_variance.astype(np.uint8), var_threshold, 255, cv2.THRESH_BINARY_INV)
    return mask
