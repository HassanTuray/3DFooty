import os
import cv2

IMAGE_DIR = r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images"
LABEL_DIR = r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\labels"

def rotate_image(image_path, rotation):
    image = cv2.imread(image_path)

    rotated_image = cv2.rotate(image, rotation)

    if rotation == cv2.ROTATE_90_CLOCKWISE:
        prefix = "rotated_90_"
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        prefix = "rotated_270_"
    elif rotation == cv2.ROTATE_180:
        prefix = "rotated_180_"
    else:
        raise ValueError("Invalid rotation")

    image_base_name = os.path.basename(image_path)
    rotated_image_path = os.path.join(IMAGE_DIR, prefix + image_base_name)

    cv2.imwrite(rotated_image_path, rotated_image)


def rotate_label(label_path, rotation):
    with open(label_path, "r") as f:
        lines = f.readlines()

    output_lines = []

    for line in lines:
        values = line.strip().split()

        if not values:
            continue

        if len(values) != 101:
            print(
                f"WARNING: {label_path} contains {len(values)} values "
                f"instead of the expected 101."
            )

        class_id = values[0]

        bbox_x = float(values[1])
        bbox_y = float(values[2])
        bbox_w = float(values[3])
        bbox_h = float(values[4])

        if rotation == cv2.ROTATE_90_CLOCKWISE:
            new_bbox_x = 1.0 - bbox_y
            new_bbox_y = bbox_x
            new_bbox_w = bbox_h
            new_bbox_h = bbox_w
            prefix = "rotated_90_"

        elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            new_bbox_x = bbox_y
            new_bbox_y = 1.0 - bbox_x
            new_bbox_w = bbox_h
            new_bbox_h = bbox_w
            prefix = "rotated_270_"

        elif rotation == cv2.ROTATE_180:
            new_bbox_x = 1.0 - bbox_x
            new_bbox_y = 1.0 - bbox_y
            new_bbox_w = bbox_w
            new_bbox_h = bbox_h
            prefix = "rotated_180_"

        else:
            raise ValueError("Invalid rotation")

        keypoints = values[5:]

        if len(keypoints) != 96:
            raise ValueError(
                f"{label_path} contains {len(keypoints)} keypoint values. "
                f"Expected 96 (32 keypoints × 3)."
            )

        rotated_keypoints = []

        for i in range(0, len(keypoints), 3):
            x = float(keypoints[i])
            y = float(keypoints[i + 1])
            visibility = keypoints[i + 2]

            if rotation == cv2.ROTATE_90_CLOCKWISE:
                new_x = 1.0 - y
                new_y = x

            elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
                new_x = y
                new_y = 1.0 - x

            elif rotation == cv2.ROTATE_180:
                new_x = 1.0 - x
                new_y = 1.0 - y

            rotated_keypoints.extend([
                f"{new_x:.6f}",
                f"{new_y:.6f}",
                visibility
            ])

        output_values = [
            class_id,
            f"{new_bbox_x:.6f}",
            f"{new_bbox_y:.6f}",
            f"{new_bbox_w:.6f}",
            f"{new_bbox_h:.6f}",
        ]

        output_values.extend(rotated_keypoints)

        if len(output_values) != 101:
            raise ValueError(
                f"Generated label has {len(output_values)} values "
                f"instead of 101."
            )

        output_lines.append(" ".join(output_values))

    original_name = os.path.basename(label_path)
    new_name = prefix + original_name

    output_path = os.path.join(LABEL_DIR, new_name)

    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    return output_path

def rotate_images():
    image_paths = os.listdir(IMAGE_DIR)

    for image_path in image_paths:
        if not ("frame" in image_path or "screenshot" in image_path or "dila" in image_path):
            rotate_image(os.path.join(IMAGE_DIR, image_path), cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotate_label(os.path.join(LABEL_DIR, image_path[:-4]+".txt"), cv2.ROTATE_90_COUNTERCLOCKWISE)
rotate_images()
