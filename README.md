# Eye Tracking Application

A terminal-based eye tracking application that accesses both front and back cameras, tracks your eyes using MediaPipe, and overlays the eye tracking visualization on the forward-facing camera feed.

## Features

- **Dual Camera Support**: Automatically detects and uses both front and back cameras
- **Real-time Eye Tracking**: Uses MediaPipe Face Mesh for accurate eye and iris tracking
- **Visual Overlay**: Displays eye landmarks, eye centers, and iris (pupil) positions
- **Camera Switching**: Switch between front and back camera views with a keypress
- **Terminal-based**: Runs from the command line

## Requirements

- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application from the terminal:

```bash
python eye_tracker.py
```

### Command Line Arguments

- `--front-camera ID`: Specify the camera ID for the front camera (default: 0)
- `--back-camera ID`: Specify the camera ID for the back camera (default: 1)

Example:
```bash
python eye_tracker.py --front-camera 0 --back-camera 1
```

### Controls

- **'q'**: Quit the application
- **'s'**: Switch between front and back camera views
- **'f'**: Toggle fullscreen mode

## Visual Indicators

- **Green circles**: Eye centers
- **Yellow dots**: Eye landmark points
- **Blue circles**: Iris (pupil) centers with outer ring
- **Yellow line**: Connection between both eyes

## Notes

- The application will automatically try to find available cameras if the default IDs don't work
- If only one camera is available, the application will still run but camera switching will be disabled
- Make sure you have proper camera permissions on your system


