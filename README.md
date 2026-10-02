# Color-Based Object Detection Using RGB and HSV

A B.Tech mini-project for detecting and comparing colored objects using
traditional RGB and HSV color-space thresholding.

## Setup

Activate the existing Python 3.11 virtual environment and install the required
dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m streamlit run app.py
```

The application supports JPG, JPEG, and PNG uploads and lets the user switch
between RGB and HSV color thresholding. Both modes provide predefined target
color ranges, manual threshold controls, and an automatically updated binary
mask. Red detection in HSV mode combines the low and high ends of OpenCV's Hue
scale. A selectable 3 x 3, 5 x 5, or 7 x 7 kernel applies morphological opening
and then closing, with the raw and cleaned masks displayed separately. External
contours from the cleaned mask are filtered by a configurable minimum area and
displayed as labeled color-region bounding boxes with an object count. An RGB
vs HSV Comparison mode sends the same image and target color through both
pipelines using shared morphology and minimum-area settings, then presents the
masks and detection results side-by-side. A separate webcam input mode provides
non-blocking live RGB or HSV detection using the same thresholding, morphology,
and contour functions. Webcam frames are converted from BGR to RGB immediately
after capture, and the camera can be started and stopped without stopping the
Streamlit server.
