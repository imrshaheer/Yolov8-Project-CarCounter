
# Yolov8-Project-CarCounter


Implemented YOLOv8 for car detection and SORT (Simple Online and Realtime Tracking) for tracking in the "Yolov8-Project-Car Detection Counter" project. Achieved car counting by monitoring their passage through a predefined region.

Data Collection: A pre-trained YOLOv8 model was used for detecting cars, along with the related car detection labels.

Data Preprocessing: The YOLOv8 model was set up to concentrate specifically on car detection. This adjustment met the project’s requirements.

Model Integration: The implementation of YOLOv8 enabled the detection of cars within video frames.

Tracking & Counting: The SORT (Simple Online & Realtime Tracking) algorithm was applied to monitor the vehicles and count how many passed through a defined area.

## Documentation

Clone this Repo

```bash
  https://github.com/imrshaheer/Yolov8-Project-CarCounter.git
```

Clone SORT Repo

```bash
  https://github.com/abewley/sort.git
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run python script

```bash
  python car-counter.py
```

