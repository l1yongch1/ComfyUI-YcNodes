import ast
import cv2
import math
import json
import torch
import numpy as np
import torch.nn.functional as F
from .utils import create_blur_mask,create_highlight_mask

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

        # 承接结果
        output_list = []
        combined_mask_list = []
        highlight_mask_list = []
        blur_mask_list = []

        for i in range(b):
            # 取第i张图片：shape [H, W, C],并将 tensor 转换为 numpy 数组，并转换形状为 [H, W, C]
            background = (background[i, :, :, :3].cpu().numpy() * 255).astype(np.uint8)

            # 如果输入图像没有 Alpha 通道，则需要另外提供或者创建一个全255的 Alpha 通道
            if c == 3:
                alpha = np.full((h, w), 255, dtype=np.uint8)
            else:
                alpha = background[:, :, 3]

            # 生成mask
            highlight_mask = create_highlight_mask(background, threshold=highlight_threshold)
            blur_mask = create_blur_mask(background, window_size=blur_window_size, var_threshold=blur_var_threshold)

            # 合并mask并进行形态学处理
            combined_mask = cv2.bitwise_or(highlight_mask, blur_mask)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

            # 更新Alpha通道
            new_alpha = cv2.bitwise_and(alpha, cv2.bitwise_not(combined_mask))

            # 加入Alpha通道
            output = np.dstack((background, new_alpha))

            # 将结果转换回 tensor, 归一化, 且转回[B, C, H, W]
            output_tensor = (torch.from_numpy(output).float() / 255.0).unsqueeze(0)
            combined_mask_tensor = (torch.from_numpy(combined_mask) / 255.0).unsqueeze(0)
            highlight_mask_tensor = (torch.from_numpy(highlight_mask) / 255.0).unsqueeze(0)
            blur_mask_tensor = (torch.from_numpy(blur_mask) / 255.0).unsqueeze(0)

            #
            output_list.append(output_tensor)
            combined_mask_list.append(combined_mask_tensor)
            highlight_mask_list.append(highlight_mask_tensor)
            blur_mask_list.append(blur_mask_tensor)

        # 合并
        output_batch = torch.cat(output_list, dim=0)
        combined_mask_batch = torch.cat(combined_mask_list, dim=0)
        highlight_mask_batch = torch.cat(highlight_mask_list, dim=0)
        blur_mask_batch = torch.cat(blur_mask_list, dim=0)

        return output_batch, combined_mask_batch, highlight_mask_batch, blur_mask_batch

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

        # 承接结果
        image_list = []
        mask_list = []

        for i in range(b):
            # 输入是[b,h,w,c]，先转numpy.array格式，
            image = (image[i, :, :, :3].cpu().numpy() * 255).astype(np.uint8)

            # 如果输入图像没有 Alpha 通道，则需要另外提供或者创建一个全255的 Alpha 通道
            if c == 3:
                mask = np.full((h, w), 255, dtype=np.uint8)
            else:
                mask = image[:, :, 3]

            # 角落坐标
            corners = [(0, 0), (w, 0), (0, h), (w, h)]

            # 在四个角绘制圆角
            for (x, y) in corners:
                center_x = (x + radius) if x + radius <= w else (x - radius)
                center_y = (y + radius) if y + radius <= h else (y - radius)

                cv2.rectangle(mask, (x, y), (center_x, center_y), 0, -1)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)

            # 把 mask 作为 Alpha 通道
            image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)  # (h, w, 4)
            image_rgba[:, :, 3] = mask

            image_rgb = image_rgba[:, :, :3]

            # numpy 2 tensor
            image_tensor = (torch.from_numpy(image_rgb).float() / 255.0).unsqueeze(0)
            mask_tensor = (torch.from_numpy(mask).float() / 255.0).unsqueeze(0)

            image_list.append(image_tensor)
            mask_list.append(mask_tensor)

        batch_output_tensor = torch.cat(image_list)
        batch_mask_tensor = torch.cat(mask_list)

        return batch_output_tensor,batch_mask_tensor

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

            # 根据图片边缘，获取背景颜色
            top_edge = current_image[0:edge_width, :, :]
            bottom_edge = current_image[-edge_width:, :, :]
            left_edge = current_image[:, 0:edge_width, :]
            right_edge = current_image[:, -edge_width:, :]

            # 计算边缘像素的均值，用于padding填充
            edge_pixels = torch.cat([
                top_edge.reshape(-1, c),
                bottom_edge.reshape(-1, c),
                left_edge.reshape(-1, c),
                right_edge.reshape(-1, c)
            ], dim=0)  # (N, C)

            pad_pixel = edge_pixels.median(dim=0).values

            # mask
            current_mask = torch.zeros_like(current_image)

            # 图片太宽了,那么应该padding长度
            if current_aspect_ratio > target_aspect_ratio:

                # 计算padding的大小
                h_target = w / target_aspect_ratio
                vertical_padding = math.ceil((h_target - h) / 2)

                # 分通道处理填充
                channels = torch.unbind(current_image, dim=-1)  # 拆分通道
                padded_channels = []

                for channel, val in zip(channels, pad_pixel):
                    # 每个通道单独填充
                    padded = F.pad(
                        channel.unsqueeze(-1),  # 保持三维形状 [H, W, 1]
                        (0, 0, vertical_padding, vertical_padding),
                        mode='constant',
                        value=val.item()  # 转为标量float
                    )
                    padded_channels.append(padded)

                # 合并通道
                padded_tensor = torch.cat(padded_channels, dim=-1)

                # mask padding
                mask_tensor = F.pad(
                     current_mask
                    ,(0,0,vertical_padding,vertical_padding)
                    ,mode="constant"
                    ,value=0
                )

            # 图片太长了，那么应该padding宽度
            elif current_aspect_ratio < target_aspect_ratio:

                # 计算padding的大小
                w_target = h * target_aspect_ratio
                horizontal_padding = math.ceil((w_target - w) / 2)

                # 分通道处理填充
                channels = torch.unbind(current_image, dim=-1)  # 拆分通道
                padded_channels = []

                for channel, val in zip(channels, pad_pixel):
                    # 每个通道单独填充
                    padded = F.pad(
                        channel.unsqueeze(-1),  # 保持三维形状 [H, W, 1]
                        (0, 0, horizontal_padding, horizontal_padding),
                        mode='constant',
                        value=val.item()  # 转为标量float
                    )
                    padded_channels.append(padded)

                # 合并通道
                padded_tensor = torch.cat(padded_channels, dim=-1)

                # mask padding
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

        return batch_image_tensor,batch_mask_tensor

NODE_CLASS_MAPPINGS = {
     'RemoveHighlightAndBlur':RemoveHighlightAndBlur
    ,'RoundedCorners':RoundedCorners
    ,'PaddingAccordingToBackground':PaddingAccordingToBackground
}

NODE_DISPLAY_NAME_MAPPINGS = {
     'RemoveHighlightAndBlur':'RemoveHighlightAndBlur'
    ,'RoundedCorners':'RoundedCorners'
    ,'PaddingAccordingToBackground':'PaddingAccordingToBackground'
}



