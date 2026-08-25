
from PIL import Image, ImageDraw
import uuid
import os

DATASET_PATH = r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset"
IMAGE_DIR = r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images"
LABEL_DIR = r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\labels"

def overlay_yolo_keypoints(image_path):
    image_name = os.path.basename(image_path)
    label_path = os.path.join(LABEL_DIR, image_name[:-4] + ".txt")

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    with open(label_path, "r") as f:
        lines = f.readlines()

    draw = ImageDraw.Draw(image)

    for line in lines:
        values = line.strip().split()

        if not values:
            continue

        bbox_x = float(values[1])
        bbox_y = float(values[2])
        bbox_w = float(values[3])
        bbox_h = float(values[4])

        bbox_left = int((bbox_x - bbox_w / 2) * width)
        bbox_top = int((bbox_y - bbox_h / 2) * height)
        bbox_right = int((bbox_x + bbox_w / 2) * width)
        bbox_bottom = int((bbox_y + bbox_h / 2) * height)

        draw.rectangle(
            (
                bbox_left,
                bbox_top,
                bbox_right,
                bbox_bottom
            ),
            outline="blue",
            width=2
        )

        keypoint_values = values[5:]

        for i in range(0, len(keypoint_values), 3):

            x = float(keypoint_values[i])
            y = float(keypoint_values[i + 1])
            visibility = int(float(keypoint_values[i + 2]))

            if visibility == 0:
                continue

            px = int(x * width)
            py = int(y * height)

            radius = 4

            draw.ellipse(
                (
                    px - radius,
                    py - radius,
                    px + radius,
                    py + radius
                ),
                fill="red"
            )

            keypoint_index = i // 3

            draw.text(
                (px + 5, py - 5),
                str(keypoint_index),
                fill="yellow"
            )

    image_name = r"C:\Users\Hassa\Coding\Projects\3dfooty\scripts\tmp_images" + f"\{uuid.uuid4()}" + ".jpg"
    image.save(image_name)

    return image

for image_path in os.listdir(IMAGE_DIR): 
    if not ("frame" in image_path or "screenshot" in image_path or "dila" in image_path) and "rotated" in image_path:
        overlay_yolo_keypoints(os.path.join(IMAGE_DIR, image_path))

# C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\08fd33_2_11.png
# C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_0_52.jpg
# C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\0a2d9b_2_12.png