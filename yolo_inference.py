from ultralytics import YOLO

# To predict the output from the trained model
model = YOLO('models/')

results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
for box in results[0].boxes:
    print(box)