# ComfyUI Custom Nodes - LYC Tool

This repository contains a collection of custom nodes for ComfyUI, designed to enhance image processing workflows. The nodes provide various functionalities including background removal, highlight and blur detection, padding, rounding corners, and AI-powered captioning.

## Nodes Overview

### 1. RemoveHighlightAndBlur
Detects and removes highlights and blurry areas from images.

**Inputs:**
- `background`: Input image
- `highlight_threshold`: Threshold for highlight detection (0-255)
- `blur_var_threshold`: Variance threshold for blur detection
- `blur_window_size`: Window size for blur detection
- `kernel_size`: Kernel size for morphological operations

**Outputs:**
- `image`: Processed image with highlights and blur removed
- `combined_mask`: Combined mask of highlights and blur
- `highlight_mask`: Mask of highlight areas
- `blur_mask`: Mask of blurry areas

### 2. RoundedCorners
Adds rounded corners to an image.

**Inputs:**
- `image`: Input image
- `radius`: Corner radius in pixels

**Outputs:**
- `image`: Image with rounded corners
- `mask`: Alpha mask showing the rounded corners

### 3. PaddingAccordingToBackground
Adds padding to an image to match a target aspect ratio, using the edge colors.

**Inputs:**
- `image`: Input image
- `target_aspect_ratio`: Desired aspect ratio
- `edge_width`: Width of edge to sample for padding color

**Outputs:**
- `image`: Padded image
- `mask`: Mask showing padded areas

### 4. QwenCaption
Generates captions for images using Qwen's vision-language models.

**Inputs:**
- `image`: Image URL to caption
- `base_url`: API base URL
- `qwen_api`: API key
- `vision_model`: Model selection (qwen2.5-vl-72b-instruct or qwen-vl-plus)
- `system_prompt`: System prompt for the model
- `user_prompt`: User prompt for the model
- `max_retries`: Maximum retry attempts

**Outputs:**
- `content_description`: Description of image content
- `content`: Additional content information
- `image`: Processed image tensor

### 5. RemoveBackground
Advanced background removal with multiple techniques.

**Inputs:**
- `image`: Input image
- `show_debug`: Whether to show debug information

**Outputs:**
- `Transparent_Image`: Image with transparent background
- `HSV_Mask`: Mask from HSV color space
- `KMeans_Mask`: Mask from KMeans clustering
- `Final_Mask`: Final refined mask

### 6. RemoveBackgroundWithProtection
Background removal with protected areas.

**Inputs:**
- `image`: Input image
- `bg_threshold`: Background color threshold
- `canny_low`: Canny edge detector low threshold
- `canny_high`: Canny edge detector high threshold
- `dilate_iter`: Dilation iterations
- `smooth_alpha`: Whether to smooth alpha channel
- `smooth_sigma`: Smoothing sigma value

**Outputs:**
- `rgba_image`: Image with transparent background
- `mask_bg`: Background mask
- `protect_mask`: Protected areas mask

### 7. RemoveBackgroundWithProtectionOptimized
Optimized version of background removal with protection.

**Inputs:**
- `image`: Input image
- `border_percent`: Percentage of border to sample
- `bg_threshold`: Background color threshold
- `canny_sigma`: Canny edge detector sigma
- `morph_iters`: Morphological operation iterations
- `smooth_alpha`: Whether to smooth alpha channel
- `smooth_radius`: Smoothing radius
- `rmbg_protect_mask`: Optional protection mask

**Outputs:**
- `rgba_image`: Image with transparent background
- `mask_bg`: Background mask
- `protect_mask`: Protected areas mask
- `final_mask`: Final combined mask

### 8. EstimateBackgroundFromTriangleCorners
Estimates background color from image corners.

**Inputs:**
- `image`: Input image
- `corner_percent`: Percentage of corner to sample

**Outputs:**
- `L`: L channel value (LAB color space)
- `A`: A channel value
- `B`: B channel value

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory.
2. Install required dependencies:
   ```
   pip install opencv-python numpy torch scikit-learn openai
   ```
3. Restart ComfyUI.

## Usage

1. Load the nodes in ComfyUI by searching for "lyc-tool" in the node menu.
2. Connect the nodes as needed in your workflow.
3. For QwenCaption, make sure to provide a valid API key and base URL.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ComfyUI for the amazing framework
- Qwen for their vision-language models
- OpenCV and scikit-learn for image processing capabilities
