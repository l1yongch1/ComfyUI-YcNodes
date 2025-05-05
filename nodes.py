import re
import math
import json
import time
import logging
import cv2
import numpy as np
import torch

from openai import OpenAI
from datetime import datetime
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .utils import create_blur_mask,create_highlight_mask,url_to_tensor

class RemoveHighlightAndBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                 'background': ('IMAGE', {})
                , 'highlight_threshold': ('INT', {'default': 150, 'min': 0, 'max': 255, 'step': 5})
                , 'blur_var_threshold': ('INT', {'default': 1, 'min': 0, 'max': 100, 'step': 1})
                , 'blur_window_size': ('INT', {'default': 25, 'min': 5, 'max': 35, 'step': 1})
                , 'kernel_size':('INT',{'default':3, 'min':1, 'max':25, 'step':1})
            }
        }

    RETURN_NAMES = ('image', 'combined_mask', 'highlight_mask', 'blur_mask')
    RETURN_TYPES = ('IMAGE', 'MASK', 'MASK', 'MASK')
    CATEGORY = 'lyc-tool'
    FUNCTION = 'get_mask'

    def get_mask(self, background, highlight_threshold, blur_var_threshold, blur_window_size, kernel_size):

        b, h, w, c = background.shape

        output_list = []
        combined_mask_list = []
        highlight_mask_list = []
        blur_mask_list = []

        for i in range(b):
            background = (background[i, :, :, :3].cpu().numpy() * 255).astype(np.uint8)

            if c == 3:
                alpha = np.full((h, w), 255, dtype=np.uint8)
            else:
                alpha = background[:, :, 3]

            highlight_mask = create_highlight_mask(background, threshold=highlight_threshold)
            blur_mask = create_blur_mask(background, window_size=blur_window_size, var_threshold=blur_var_threshold)

            combined_mask = cv2.bitwise_or(highlight_mask, blur_mask)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            new_alpha = cv2.bitwise_and(alpha, cv2.bitwise_not(combined_mask))

            output = np.dstack((background, new_alpha))

            output_tensor = (torch.from_numpy(output).float() / 255.0).unsqueeze(0)
            combined_mask_tensor = (torch.from_numpy(combined_mask) / 255.0).unsqueeze(0)
            highlight_mask_tensor = (torch.from_numpy(highlight_mask) / 255.0).unsqueeze(0)
            blur_mask_tensor = (torch.from_numpy(blur_mask) / 255.0).unsqueeze(0)

            output_list.append(output_tensor)
            combined_mask_list.append(combined_mask_tensor)
            highlight_mask_list.append(highlight_mask_tensor)
            blur_mask_list.append(blur_mask_tensor)

        output_batch = torch.cat(output_list, dim=0)
        combined_mask_batch = torch.cat(combined_mask_list, dim=0)
        highlight_mask_batch = torch.cat(highlight_mask_list, dim=0)
        blur_mask_batch = torch.cat(blur_mask_list, dim=0)

        return (output_batch, combined_mask_batch, highlight_mask_batch, blur_mask_batch)

class RoundedCorners:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required':{
                 'image':('IMAGE',{})
                ,'radius':('INT',{'default':100,'max':9999, 'min':0, 'step':1})
            }
        }

    RETURN_NAMES = ('image','mask')
    RETURN_TYPES = ('IMAGE','MASK')
    CATEGORY = 'lyc-tool'
    FUNCTION = 'corner'

    def corner(self, image, radius):

        b, h, w, c = image.shape

        image_list = []
        mask_list = []

        for i in range(b):
            image = (image[i, :, :, :3].cpu().numpy() * 255).astype(np.uint8)

            if c == 3:
                mask = np.full((h, w), 255, dtype=np.uint8)
            else:
                mask = image[:, :, 3]

            corners = [(0, 0), (w, 0), (0, h), (w, h)]

            for (x, y) in corners:
                center_x = (x + radius) if x + radius <= w else (x - radius)
                center_y = (y + radius) if y + radius <= h else (y - radius)

                cv2.rectangle(mask, (x, y), (center_x, center_y), 0, -1)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)

            image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)  # (h, w, 4)
            image_rgba[:, :, 3] = mask

            image_rgb = image_rgba[:, :, :3]

            image_tensor = (torch.from_numpy(image_rgb).float() / 255.0).unsqueeze(0)
            mask_tensor = (torch.from_numpy(mask).float() / 255.0).unsqueeze(0)

            image_list.append(image_tensor)
            mask_list.append(mask_tensor)

        batch_output_tensor = torch.cat(image_list)
        batch_mask_tensor = torch.cat(mask_list)

        return (batch_output_tensor,batch_mask_tensor)

class PaddingAccordingToBackground:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required':{
                 'image':('IMAGE', {})
                ,'target_aspect_ratio':('FLOAT', {'default':1.0, 'max':5.00, 'min':0.01, 'step':0.01})
                ,'edge_width':('INT', {'default':10, 'max':9999, 'min':1, 'step':1})
            }
        }

    RETURN_NAMES = ('image', 'mask')
    RETURN_TYPES = ('IMAGE', 'MASK')
    CATEGORY = 'lyc-tool'
    FUNCTION = 'padding'

    def padding(self,image,target_aspect_ratio,edge_width):

        b,h,w,c = image.shape
        image_list = []
        mask_list = []

        for i in range(b):
            current_image = image[i, :, :, :]
            current_aspect_ratio = w/h

            top_edge = current_image[0:edge_width, :, :]
            bottom_edge = current_image[-edge_width:, :, :]
            left_edge = current_image[:, 0:edge_width, :]
            right_edge = current_image[:, -edge_width:, :]

            edge_pixels = torch.cat([
                top_edge.reshape(-1, c),
                bottom_edge.reshape(-1, c),
                left_edge.reshape(-1, c),
                right_edge.reshape(-1, c)
            ], dim=0)  # (N, C)

            pad_pixel = edge_pixels.median(dim=0).values

            current_mask = torch.zeros_like(current_image)

            if current_aspect_ratio > target_aspect_ratio:

                h_target = w / target_aspect_ratio
                vertical_padding = math.ceil((h_target - h) / 2)

                channels = torch.unbind(current_image, dim=-1)
                padded_channels = []

                for channel, val in zip(channels, pad_pixel):
                    padded = F.pad(
                        channel.unsqueeze(-1),
                        (0, 0, vertical_padding, vertical_padding),
                        mode='constant',
                        value=val.item()
                    )
                    padded_channels.append(padded)

                padded_tensor = torch.cat(padded_channels, dim=-1)

                mask_tensor = F.pad(
                     current_mask
                    ,(0,0,vertical_padding,vertical_padding)
                    ,mode="constant"
                    ,value=0
                )

            elif current_aspect_ratio < target_aspect_ratio:

                w_target = h * target_aspect_ratio
                horizontal_padding = math.ceil((w_target - w) / 2)

                channels = torch.unbind(current_image, dim=-1)  # 拆分通道
                padded_channels = []

                for channel, val in zip(channels, pad_pixel):
                    padded = F.pad(
                        channel.unsqueeze(-1),
                        (0, 0, horizontal_padding, horizontal_padding),
                        mode='constant',
                        value=val.item()
                    )
                    padded_channels.append(padded)

                padded_tensor = torch.cat(padded_channels, dim=-1)

                mask_tensor = F.pad(
                     current_mask
                    ,(0,0,horizontal_padding,horizontal_padding)
                    ,mode="constant"
                    ,value=0
                )

            else:
                padded_tensor = current_image
                mask_tensor = current_mask

            image_list.append(padded_tensor.unsqueeze(0))
            mask_list.append(mask_tensor.unsqueeze(0))

        batch_image_tensor = torch.cat(image_list)
        batch_mask_tensor = torch.cat(mask_list)

        return (batch_image_tensor,batch_mask_tensor)

class QwenCaption:
    model_list = [
         'qwen2.5-vl-72b-instruct'
        ,'qwen-vl-plus'
    ]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            'required':{
                 'image':('STRING', {'default':'default_url'})
                ,'base_url':('STRING', {'default':'https://dashscope.aliyuncs.com/compatible-mode/v1'})
                ,'qwen_api':('STRING', {})
                ,'vision_model':(self.model_list,)
                ,'system_prompt':('STRING', {'multiline':True})
                ,'user_prompt':('STRING', {'multiline':True})
                ,'max_retries':('INT', {'default': 3, 'max': 5, 'min': 0, 'step': 1})
            }
        }

    RETURN_TYPES = ('STRING','STRING','IMAGE')
    RETURN_NAMES = ('content_description', 'content', 'image')
    CATEGORY = 'lyc-tool'
    FUNCTION = 'caption'

    def caption(self, image, base_url, qwen_api, vision_model, system_prompt, user_prompt, max_retries):

        # image
        image_tensor = url_to_tensor(image)

        qwen_client = OpenAI(
            base_url=base_url,
            api_key=qwen_api,
        )

        attempts = 0
        stop = False
        content_description = ''
        content = ''

        while not stop and attempts < max_retries:
            try:
                start_time = datetime.now()
                completion = qwen_client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "system"
                            , "content": system_prompt
                        },

                        {
                            "role": "user"
                            , "content": [
                              {"type": "text", "text": user_prompt}
                             ,{"type": "image_url", "image_url": {"url": image}}
                            ]
                        }
                    ]
                )
                end_time = datetime.now()

                time_diff = end_time - start_time
                print("相差的总秒数:", time_diff.total_seconds())

                # 获取返回的内容
                content = json.loads(completion.model_dump_json())
                content = content['choices'][0]['message']['content']

                # 清除Markdown格式的标记
                content = re.sub(r"[*#]", "", content)
                content_dict = json.loads(content)
                content_description = content_dict.get('Content Description','')
                content = content_dict.get('Content','')

                # print('Qwen 调用完成')
                stop = True

            except Exception as e:
                logging.info(f'调用Qwen时报错 {e}')
                attempts += 1
                time.sleep(1)
                content_description = ''
                content = ''

        return content_description,content,image_tensor

class RemoveBackground:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # tensor: (1, H, W, 3), float32, [0,1]
                "show_debug": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("Transparent_Image", "HSV_Mask", "KMeans_Mask", "Final_Mask")
    FUNCTION = "process"
    CATEGORY = "Background Removal"

    def process(self, image, show_debug=True):
        # === Step 1: 读取图像 ===
        image_np = image[0].cpu().numpy()              # (H, W, 3), float32
        image_np = (image_np * 255).astype(np.uint8)   # to uint8 [0,255]
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]

        # === Step 2: HSV 分割白背景 ===
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

        # === Step 3: KMeans 聚类 ===
        Z = image_bgr.reshape((-1, 3)).astype(np.float32)
        kmeans = KMeans(n_clusters=2, random_state=42).fit(Z)
        labels = kmeans.labels_.reshape(h, w)
        bg_label = 0 if np.sum(labels == 0) > np.sum(labels == 1) else 1
        mask_kmeans = (labels != bg_label).astype(np.uint8) * 255

        # === Step 4: 合并 mask ===
        combined = np.maximum(mask_hsv, mask_kmeans)

        # === Step 5: GrabCut 精修 ===
        mask = np.zeros((h, w), np.uint8)
        mask[combined > 0] = cv2.GC_PR_FGD
        mask[combined == 0] = cv2.GC_PR_BGD
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(image_bgr, mask, None, bgModel, fgModel, 5, cv2.GC_INIT_WITH_MASK)
        result_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8) * 255

        # === Step 6: 后处理 ===
        kernel = np.ones((3, 3), np.uint8)
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel)
        result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)

        # === Step 7: 添加透明通道（RGBA）===
        # 将BGR转回RGB
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # 直接创建RGBA数组
        result_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        result_rgba[:, :, :3] = rgb_image  # RGB通道
        result_rgba[:, :, 3] = result_mask  # Alpha通道

        # 转换为float32并归一化
        rgba_np = result_rgba.astype(np.float32) / 255.0

        # === Step 8: 转为 PyTorch Tensor ===
        rgba_tensor = torch.from_numpy(rgba_np).unsqueeze(0)  # (1, H, W, 4)
        mask_hsv_tensor = torch.from_numpy(mask_hsv.astype(np.float32) / 255.0)  # (H, W)
        mask_kmeans_tensor = torch.from_numpy(mask_kmeans.astype(np.float32) / 255.0)
        result_mask_tensor = torch.from_numpy(result_mask.astype(np.float32) / 255.0)

        return (rgba_tensor, mask_hsv_tensor, mask_kmeans_tensor, result_mask_tensor)

class RemoveBackgroundWithProtection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bg_threshold": ("INT", {"default": 20, "min": 1, "max": 100}),
                "canny_low": ("INT", {"default": 100, "min": 0, "max": 255}),
                "canny_high": ("INT", {"default": 200, "min": 0, "max": 255}),
                "dilate_iter": ("INT", {"default": 2, "min": 0, "max": 10}),
                "smooth_alpha": ("BOOLEAN", {"default": True}),
                "smooth_sigma": ("INT", {"default": 3, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("rgba_image", "mask_bg", "protect_mask")
    FUNCTION = "process"
    CATEGORY = "image/processing"

    def estimate_background_color(self, image_bgr, border_width=20):
        top = image_bgr[:border_width, :, :]
        bottom = image_bgr[-border_width:, :, :]
        left = image_bgr[:, :border_width, :]
        right = image_bgr[:, -border_width:, :]
        border_pixels = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ], axis=0)
        kmeans = KMeans(n_clusters=2, random_state=42).fit(border_pixels.astype(np.float32))
        counts = np.bincount(kmeans.labels_)
        bg_color = kmeans.cluster_centers_[np.argmax(counts)]
        return bg_color.astype(np.uint8)

    def extract_filled_protect_mask(self, image_bgr, canny_low, canny_high, dilation_iter):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=dilation_iter)
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        protect_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(protect_mask, contours, contourIdx=-1, color=255, thickness=-1)
        return protect_mask

    def process(self, image, bg_threshold, canny_low, canny_high, dilate_iter, smooth_alpha, smooth_sigma):
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]

        # 背景主色估计
        bg_color = self.estimate_background_color(image_bgr)

        # 背景颜色差 mask
        diff = np.linalg.norm(image_bgr.astype(np.float32) - bg_color[None, None, :], axis=2)
        mask_bg = (diff < bg_threshold).astype(np.uint8) * 255

        # 保护区域 mask
        protect_mask = self.extract_filled_protect_mask(image_bgr, canny_low, canny_high, dilate_iter)

        # 合并：去除背景，但保留保护区域
        final_mask = mask_bg.copy()
        final_mask[protect_mask > 0] = 0  # 保护区域保留

        # 可选：边缘平滑处理
        if smooth_alpha:
            final_mask = cv2.GaussianBlur(final_mask, (0, 0), sigmaX=smooth_sigma, sigmaY=smooth_sigma)
            final_mask = np.clip(final_mask, 0, 255).astype(np.uint8)

        # 构建 RGBA 输出
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        result_rgba[:, :, :3] = rgb_image
        result_rgba[:, :, 3] = 255 - final_mask  # 背景区域透明

        # 转为 tensor 输出
        rgba_tensor = torch.from_numpy(result_rgba.astype(np.float32) / 255.0).unsqueeze(0)
        mask_bg_tensor = torch.from_numpy(mask_bg.astype(np.float32) / 255.0)
        protect_mask_tensor = torch.from_numpy(protect_mask.astype(np.float32) / 255.0)

        print(rgba_tensor.shape)
        print(mask_bg_tensor.shape)
        print(protect_mask_tensor.shape)

        return (rgba_tensor, mask_bg_tensor, protect_mask_tensor)

class RemoveBackgroundWithProtectionOptimized:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_percent":("INT",{"default": 3, "min": 1, "max": 100, "step":1}),
                "bg_threshold": ("INT", {"default": 15, "min": 1, "max": 100}),
                "canny_sigma": ("FLOAT", {"default": 0.33, "min": 0.1, "max": 1.0}),
                "morph_iters": ("INT", {"default": 2, "min": 0, "max": 5}),
                "smooth_alpha": ("BOOLEAN", {"default": True}),
                "smooth_radius": ("INT", {"default": 5, "min": 1, "max": 20}),
            }

            ,"optional":{
                "rmbg_protect_mask":("MASK", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("rgba_image", "mask_bg", "protect_mask", "final_mask")
    FUNCTION = "process"
    CATEGORY = "lyc-tool"

    # 四边判定
    def estimate_background_edge(self, img_bgr, border_percent=3):
        h, w = img_bgr.shape[:2]
        border_size = int(min(h, w) * border_percent / 100)

        # 正确版本：边缘设为1，内区设为0
        border_mask = np.ones((h, w), dtype=np.uint8)
        cv2.rectangle(border_mask,
                      (border_size, border_size),
                      (w - border_size, h - border_size),
                      0, -1)

        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        border_pixels = img_lab[border_mask == 1]

        if border_pixels.size == 0:
            raise ValueError("Background mask is empty. Check border size settings.")

        dominant_color = np.median(border_pixels, axis=0)
        return dominant_color.astype(np.uint8)

    # 四个角判定
    def estimate_background_corner(self, img_bgr, border_percent=3):

        h, w = img_bgr.shape[:2]
        max_corner_h = int(h * border_percent / 100)
        max_corner_w = int(w * border_percent / 100)

        # 创建掩码
        mask = np.zeros((h, w), dtype=np.uint8)

        for y in range(max_corner_h):
            for x in range(max_corner_w):
                if x + y <= max_corner_w:
                    mask[y, x] = 1  # 左上

        for y in range(max_corner_h):
            for x in range(w - max_corner_w, w):
                if (w - x - 1) + y <= max_corner_w:
                    mask[y, x] = 1  # 右上

        for y in range(h - max_corner_h, h):
            for x in range(max_corner_w):
                if x + (h - y - 1) <= max_corner_w:
                    mask[y, x] = 1  # 左下

        for y in range(h - max_corner_h, h):
            for x in range(w - max_corner_w, w):
                if (w - x - 1) + (h - y - 1) <= max_corner_w:
                    mask[y, x] = 1  # 右下

        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        corner_pixels = img_lab[mask == 1]

        if corner_pixels.size == 0:
            raise ValueError("No corner pixels found. Try increasing corner_percent.")

        dominant_color = np.median(corner_pixels, axis=0).astype(np.uint8)
        return dominant_color.astype(np.uint8)

    def auto_canny(self, gray, sigma=0.33):
        v = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(gray, lower, upper)

    def refine_protection_mask(self, edges, iterations=2):
        # 形态学闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        # 孔洞填充
        floodfill = closed.copy()
        h, w = closed.shape[:2]
        cv2.floodFill(floodfill, None, (0, 0), 255)
        return closed | cv2.bitwise_not(floodfill)

    def process(self, image, border_percent, bg_threshold, canny_sigma, morph_iters, smooth_alpha, smooth_radius, rmbg_protect_mask=None):
        img = image[0,:,:,]
        # 转换到numpy格式
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)[..., :3]
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # 步骤1: 估计背景颜色（LAB空间）
        bg_color_lab = self.estimate_background_corner(img_bgr,border_percent)

        # 步骤2: 计算颜色差异（LAB空间）
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        diff = np.linalg.norm(img_lab.astype(np.float32) - bg_color_lab.astype(np.float32), axis=2)

        # 步骤3: 根据背景阈值识别背景
        mask_bg = (diff < bg_threshold).astype(np.uint8) * 255

        # 步骤4: 生成保护区域
        if rmbg_protect_mask is None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            edges = self.auto_canny(gray, canny_sigma)
            protect_mask = self.refine_protection_mask(edges, morph_iters)
        else:
            # protect_mask = (rmbg_protect_mask[0].cpu().numpy() * 255).astype(np.uint8)
            if rmbg_protect_mask.ndim == 3:
                protect_mask = rmbg_protect_mask[0].cpu().numpy()
                print(protect_mask.shape)
            else:
                protect_mask = rmbg_protect_mask.cpu().numpy()
                print(protect_mask.shape)

        protect_mask = (protect_mask * 255).astype(np.uint8)

        # 合并：去除背景，但保留保护区域
        final_mask = mask_bg.copy()
        final_mask[protect_mask > 0] = 0  # 保护区域保留

        # 步骤6: Alpha平滑处理
        if smooth_alpha:
            final_mask = cv2.GaussianBlur(final_mask,
                                          (2 * smooth_radius + 1, 2 * smooth_radius + 1),
                                          0)
            final_mask = np.clip(final_mask, 0, 255).astype(np.uint8)

        # 构建RGBA图像
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgba[..., 3] = 255 - final_mask

        rgba = torch.from_numpy(rgba.astype(np.float32) / 255).unsqueeze(0)
        mask_bg = torch.from_numpy(mask_bg.astype(np.float32) / 255)
        protect_mask = torch.from_numpy(protect_mask.astype(np.float32) / 255)
        final_mask = torch.from_numpy(final_mask.astype(np.float32) / 255)

        return (rgba,mask_bg,protect_mask,final_mask)

class EstimateBackgroundFromTriangleCorners:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # b,h,w,c tensor
                "corner_percent": ("INT", {"default": 3, "min": 1, "max": 30}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("L", "A", "B")
    FUNCTION = "estimate"

    CATEGORY = "lyc-tool"

    def estimate(self, image, corner_percent):
        img_bgr = (image[0].cpu().numpy() * 255).astype(np.uint8)  # h,w,c
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)


        h, w = img_bgr.shape[:2]
        max_corner_h = int(h * corner_percent / 100)
        max_corner_w = int(w * corner_percent / 100)

        # 创建掩码
        mask = np.zeros((h, w), dtype=np.uint8)

        for y in range(max_corner_h):
            for x in range(max_corner_w):
                if x + y <= max_corner_w:
                    mask[y, x] = 1  # 左上

        for y in range(max_corner_h):
            for x in range(w - max_corner_w, w):
                if (w - x - 1) + y <= max_corner_w:
                    mask[y, x] = 1  # 右上

        for y in range(h - max_corner_h, h):
            for x in range(max_corner_w):
                if x + (h - y - 1) <= max_corner_w:
                    mask[y, x] = 1  # 左下

        for y in range(h - max_corner_h, h):
            for x in range(w - max_corner_w, w):
                if (w - x - 1) + (h - y - 1) <= max_corner_w:
                    mask[y, x] = 1  # 右下

        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        corner_pixels = img_lab[mask == 1]

        if corner_pixels.size == 0:
            raise ValueError("No corner pixels found. Try increasing corner_percent.")

        dominant_color = np.median(corner_pixels, axis=0).astype(np.uint8)
        L, A, B = int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])

        return (L, A, B)




