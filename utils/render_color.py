import cv2
import numpy as np


def render_input(msi_image: np.ndarray):
    assert msi_image.ndim == 3
    # C, H, W   ----->    H, W, 3 for display
    R = (255 * msi_image[8, ...]).astype(np.uint8)  # 8
    G = (255 * msi_image[8, ...]).astype(np.uint8)
    B = (255 * msi_image[8, ...]).astype(np.uint8)
    input_img = cv2.merge([B, G, R])
    return input_img


def render_output(pred_label: np.ndarray):
    # 定义RGB颜色映射表，每个类别对应的颜色
    color_map = {
        0: [0, 0, 0],   # 类别 0 映射为纯黑色
        1: [0, 0, 255],  # 类别 1 映射为纯红色
        2: [0, 255, 0],  # 类别 1 映射为纯绿色
        3: [255, 0, 255]  # 类别 2 映射为纯紫色
    }
    height, width = pred_label.shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    for label, color in color_map.items():
        output_image[pred_label == label] = color
    return output_image


def render_stacked(back, gas):
    gas_region = (255 * gas.astype("<f4").sum(axis=2) > 0).astype(np.uint8)
    gas_border = cv2.subtract(gas_region, cv2.erode(gas_region, np.ones((3, 3), dtype=np.uint8), iterations=1)) > 0
    no_gas_region = gas.astype("<f4").sum(axis=2) == 0
    B = back[:, :, 0] * (170.0 / 255.0) + 80 * (gas[:, :, 0] / 255.0)
    G = back[:, :, 1] * (170.0 / 255.0) + 80 * (gas[:, :, 1] / 255.0)
    R = back[:, :, 2] * (170.0 / 255.0) + 80 * (gas[:, :, 2] / 255.0)
    B[no_gas_region] = back[:, :, 0][no_gas_region]
    G[no_gas_region] = back[:, :, 1][no_gas_region]
    R[no_gas_region] = back[:, :, 2][no_gas_region]
    B[gas_border] = gas[:, :, 0].max()
    G[gas_border] = gas[:, :, 1].max()
    R[gas_border] = gas[:, :, 2].max()
    return cv2.merge([B.astype(np.uint8), G.astype(np.uint8), R.astype(np.uint8)])

