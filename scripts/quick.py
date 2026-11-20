import sys
import cv2

NAMES = ["TLC","L18BYTC","L6BYTC","L6BYBC","L18BYBC","BLC","L6TRC","L6BRC","LPK",
         "L18TC","L18TPKA","L18BPKA","L18BC","HTT","HCT","HCB","HBT",
         "R18TC","R18TPKA","R18BPKA","R18BC","RPK","R6TRC","R6BRC",
         "TRC","R18BYTC","R6BYTC","R6BYBC","R18BYBC","BRC","HCL","HCR"]

KPTS = [
 (47,45),(47,92),(47,145),(47,233),(47,286),(47,333),
 (73,145),(73,233),(99,189),
 (126,92),(126,153),(126,225),(126,286),
 (263,45),(263,145),(263,233),(263,333),
 (401,92),(401,153),(401,225),(401,286),
 (427,189),(454,145),(454,233),
 (480,45),(480,92),(480,145),(480,233),(480,286),(480,333),
 (219,189),(307,189),
]

def main():
    if len(sys.argv) < 2:
        print("usage: python overlay_kpts.py <image> [output]")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "overlay.png"

    img = cv2.imread(src)
    if img is None:
        sys.exit(f"could not read {src}")

    for i, (x, y) in enumerate(KPTS):
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(img, f"{i}:{NAMES[i]}", (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(out, img)
    print(f"saved {out}")
    cv2.imshow("keypoints", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()