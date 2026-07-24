# RescueEye

RescueEye is a computer vision prototype that analyses recorded drone footage and detects rescue-relevant objects.

I built this project after attending an FPV drone workshop organised by **Brobot**. The workshop made me think about how drone footage could be used for practical applications beyond flying and recording videos.

## What RescueEye Does

RescueEye processes drone footage frame by frame using YOLO for object detection and ByteTrack for object tracking.

The system can:

- Detect selected objects in drone footage
- Assign tracking IDs to detected objects
- Display confidence scores
- Divide the footage into search sectors
- Save evidence screenshots
- Flag low visible movement in detected people
- Generate a CSV detection log
- Generate a PDF mission report
- Create an annotated output video

## Detection Modes

The application includes several detection profiles:

### Rescue Essentials

Detects:

- People
- Cars
- Trucks
- Buses
- Motorcycles
- Bicycles
- Boats
- Aircraft
- Common animals
- Backpacks
- Handbags
- Suitcases
- Umbrellas
- Traffic signs

### People Only

Detects and tracks only people.

### People and Vehicles

Detects people, cars, buses, trucks, motorcycles, bicycles, boats and aircraft.

### People, Animals and Bags

Detects people, common animals, backpacks, handbags, umbrellas and suitcases.

### All Supported Objects

Enables all object classes supported by the pretrained YOLO model.

### Custom Selection

Allows the user to manually choose which objects should be detected.

## How It Works

1. The user uploads a drone video.
2. OpenCV reads the video frame by frame.
3. YOLO detects selected objects in each processed frame.
4. ByteTrack attempts to maintain the same tracking ID across frames.
5. RescueEye records the object type, confidence score, timestamp and screen sector.
6. Evidence frames are saved when a new object is detected.
7. Low apparent movement alerts are calculated for people.
8. The application generates a mission report and detection log.

## Search Sector System

The video frame is divided into nine sectors:

```text
A1  B1  C1
A2  B2  C2
A3  B3  C3
