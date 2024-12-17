# Traffic Light Detection

![Traffic Light Detection Banner]

## Overview

This project is a real-time traffic light detection system implemented in Python using OpenCV. It detects red, yellow, and green lights from video streams and logs detections with timestamps.

## Features

- Real-time detection of traffic light colors (red, yellow, green)
- Adjustable HSV thresholds for color filtering using trackbars
- Logs detected colors with timestamps to a text file
- Displays processed frames in HSV, grayscale, and annotated formats

## Prerequisites

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dansecret/Trafic_Light_Detection.git
   cd Trafic_Light_Detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the program with your video file:
   ```bash
   python main.py
   ```

## Usage

1. Open the `main.py` file and set the `video_path` variable to your video file path.
2. Adjust HSV thresholds using the trackbars to fine-tune the color detection for your video.
3. Press `q` to exit the program.

## Screenshots

### Detection in Action
![Detection Screenshot](https://github.com/dansecret/Trafic_Light_Detection/raw/main/screenshoot/image.png)

### HSV Threshold Adjustment
![HSV Threshold Adjustment](https://github.com/dansecret/Trafic_Light_Detection/raw/main/screenshoot/trackbars.png)

## Demo Video

[![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://github.com/dansecret/Trafic_Light_Detection/raw/main/video/Detection.mp4)

## File Structure

```
Trafic_Light_Detection/
|-- main.py
|-- detections.txt
|-- requirements.txt
|-- screenshots/
|   |-- banner.png
|   |-- detection.png
|   |-- trackbars.png
|-- video/
    |-- demo.mp4
```

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue to discuss improvements or report bugs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

Special thanks to the OpenCV community for providing robust libraries for image processing.