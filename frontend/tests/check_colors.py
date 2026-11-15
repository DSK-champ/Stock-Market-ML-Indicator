from PIL import Image
import numpy as np

img = Image.open(r"C:\Users\SRAVAN\.gemini\antigravity\brain\c99e21cb-57dc-4239-9f0a-bc67ef4be949\index_chart_2401cs45_indicator.png")
print("Image size:", img.size)
arr = np.array(img)

# Check for our colors:
# Blue #2962ff -> (41, 98, 255)
# Red #ef5350 -> (239, 83, 80)
# Green #26a69a -> (38, 166, 154)

blue_mask = (abs(arr[:, :, 0] - 41) < 15) & (abs(arr[:, :, 1] - 98) < 15) & (abs(arr[:, :, 2] - 255) < 15)
red_mask = (abs(arr[:, :, 0] - 239) < 15) & (abs(arr[:, :, 1] - 83) < 15) & (abs(arr[:, :, 2] - 80) < 15)
green_mask = (abs(arr[:, :, 0] - 38) < 15) & (abs(arr[:, :, 1] - 166) < 15) & (abs(arr[:, :, 2] - 154) < 15)
orange_mask = (abs(arr[:, :, 0] - 255) < 15) & (abs(arr[:, :, 1] - 152) < 25) & (abs(arr[:, :, 2] - 0) < 25)

print("Blue pixels (#2962ff):", np.sum(blue_mask))
print("Red pixels (#ef5350):", np.sum(red_mask))
print("Green pixels (#26a69a):", np.sum(green_mask))
print("Orange pixels (#ff9800):", np.sum(orange_mask))
