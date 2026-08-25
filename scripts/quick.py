from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

path = r"C:\Users\Hassa\Coding\Projects\3dfooty\professional-photo_jpg.rf.3f23f0e66d030af1a3d2d534dff4c7b4.jpg"

results = model(path)

for result in results:
    result.show()