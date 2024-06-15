#Importing packages
from ultralytics import YOLO

#Loading the model
model=YOLO('models/best.pt')


#Running the YOLO model
results = model.predict('input_videos/FA_Cup_2024.mp4',save=True)

#Show the results
print(results[0])

## Show the results of a bounding box
print("#################################################")
for box in results[0].boxes:
    print(box)