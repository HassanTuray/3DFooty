from ultralytics import YOLO

model = YOLO(r"C:\Users\Hassa\Coding\Projects\3dfooty\runs\pose\model1-6\weights\best.pt")

results = model(r"C:\Users\Hassa\Coding\Projects\3dfooty\birds_eye_view_field.png")

for result in results:
    result.show()