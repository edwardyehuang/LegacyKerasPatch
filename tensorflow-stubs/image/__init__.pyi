"""Type stubs for tensorflow.image module - provided by LegacyKerasPatch."""

from typing import Any

from tensorflow.python.ops.image_ops_impl import ResizeMethod as ResizeMethod

from tensorflow.python.ops.image_ops_impl import adjust_brightness as adjust_brightness
from tensorflow.python.ops.image_ops_impl import adjust_contrast as adjust_contrast
from tensorflow.python.ops.image_ops_impl import adjust_gamma as adjust_gamma
from tensorflow.python.ops.image_ops_impl import adjust_hue as adjust_hue
from tensorflow.python.ops.image_ops_impl import adjust_jpeg_quality as adjust_jpeg_quality
from tensorflow.python.ops.image_ops_impl import adjust_saturation as adjust_saturation
from tensorflow.python.ops.image_ops_impl import central_crop as central_crop
from tensorflow.python.ops.image_ops_impl import convert_image_dtype as convert_image_dtype
from tensorflow.python.ops.image_ops_impl import crop_and_resize_v2 as crop_and_resize
from tensorflow.python.ops.image_ops_impl import crop_to_bounding_box as crop_to_bounding_box
from tensorflow.python.ops.gen_image_ops import decode_bmp as decode_bmp
from tensorflow.python.ops.gen_image_ops import decode_gif as decode_gif
from tensorflow.python.ops.image_ops_impl import decode_image as decode_image
from tensorflow.python.ops.gen_image_ops import decode_jpeg as decode_jpeg
from tensorflow.python.ops.gen_image_ops import decode_png as decode_png
from tensorflow.python.ops.image_ops_impl import draw_bounding_boxes_v2 as draw_bounding_boxes
from tensorflow.python.ops.gen_image_ops import encode_jpeg as encode_jpeg
from tensorflow.python.ops.gen_image_ops import encode_png as encode_png
from tensorflow.python.ops.image_ops_impl import extract_patches as extract_patches
from tensorflow.python.ops.image_ops_impl import flip_left_right as flip_left_right
from tensorflow.python.ops.image_ops_impl import flip_up_down as flip_up_down
from tensorflow.python.ops.image_ops_impl import grayscale_to_rgb as grayscale_to_rgb
from tensorflow.python.ops.gen_image_ops import hsv_to_rgb as hsv_to_rgb
from tensorflow.python.ops.image_ops_impl import non_max_suppression_v2 as non_max_suppression
from tensorflow.python.ops.image_ops_impl import pad_to_bounding_box as pad_to_bounding_box
from tensorflow.python.ops.image_ops_impl import per_image_standardization as per_image_standardization
from tensorflow.python.ops.image_ops_impl import psnr as psnr
from tensorflow.python.ops.image_ops_impl import random_brightness as random_brightness
from tensorflow.python.ops.image_ops_impl import random_contrast as random_contrast
from tensorflow.python.ops.image_ops_impl import random_crop as random_crop
from tensorflow.python.ops.image_ops_impl import random_flip_left_right as random_flip_left_right
from tensorflow.python.ops.image_ops_impl import random_flip_up_down as random_flip_up_down
from tensorflow.python.ops.image_ops_impl import random_hue as random_hue
from tensorflow.python.ops.image_ops_impl import random_jpeg_quality as random_jpeg_quality
from tensorflow.python.ops.image_ops_impl import random_saturation as random_saturation
from tensorflow.python.ops.image_ops_impl import resize_images_v2 as resize
from tensorflow.python.ops.image_ops_impl import resize_image_with_crop_or_pad as resize_with_crop_or_pad
from tensorflow.python.ops.image_ops_impl import resize_image_with_pad_v2 as resize_with_pad
from tensorflow.python.ops.image_ops_impl import rgb_to_grayscale as rgb_to_grayscale
from tensorflow.python.ops.gen_image_ops import rgb_to_hsv as rgb_to_hsv
from tensorflow.python.ops.image_ops_impl import rgb_to_yiq as rgb_to_yiq
from tensorflow.python.ops.image_ops_impl import rgb_to_yuv as rgb_to_yuv
from tensorflow.python.ops.image_ops_impl import rot90 as rot90
from tensorflow.python.ops.image_ops_impl import sample_distorted_bounding_box_v2 as sample_distorted_bounding_box
from tensorflow.python.ops.image_ops_impl import sobel_edges as sobel_edges
from tensorflow.python.ops.image_ops_impl import ssim as ssim
from tensorflow.python.ops.image_ops_impl import ssim_multiscale as ssim_multiscale
from tensorflow.python.ops.image_ops_impl import stateless_random_brightness as stateless_random_brightness
from tensorflow.python.ops.image_ops_impl import stateless_random_contrast as stateless_random_contrast
from tensorflow.python.ops.image_ops_impl import stateless_random_crop as stateless_random_crop
from tensorflow.python.ops.image_ops_impl import stateless_random_flip_left_right as stateless_random_flip_left_right
from tensorflow.python.ops.image_ops_impl import stateless_random_flip_up_down as stateless_random_flip_up_down
from tensorflow.python.ops.image_ops_impl import stateless_random_hue as stateless_random_hue
from tensorflow.python.ops.image_ops_impl import stateless_random_jpeg_quality as stateless_random_jpeg_quality
from tensorflow.python.ops.image_ops_impl import stateless_random_saturation as stateless_random_saturation
from tensorflow.python.ops.image_ops_impl import stateless_sample_distorted_bounding_box as stateless_sample_distorted_bounding_box
from tensorflow.python.ops.image_ops_impl import total_variation as total_variation
from tensorflow.python.ops.image_ops_impl import transpose_v2 as transpose
from tensorflow.python.ops.image_ops_impl import yiq_to_rgb as yiq_to_rgb
from tensorflow.python.ops.image_ops_impl import yuv_to_rgb as yuv_to_rgb

def __getattr__(name: str) -> Any: ...
