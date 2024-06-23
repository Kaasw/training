from ultralytics import YOLO

model = YOLO('Trained/runs/classify/train/weights/best.pt')  # Load model

# Run inference on a single image
results = model('https://t4.ftcdn.net/jpg/04/33/82/17/360_F_433821742_0XD7pGKmNXfcuLqMV6yhrppZy7gEX8uN.jpg')
results.show()
