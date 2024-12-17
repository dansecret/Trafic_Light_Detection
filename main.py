import cv2
import numpy as np

# Konstanta warna (HSV Range)
COLOR_RANGES = {
    "Red": ((102, 0, 57), (189, 255, 255), (0, 0, 255)),
    "Yellow": ((14, 0, 154), (42, 255, 255), (0, 255, 255)),
    "Green": ((82, 0, 156), (123, 255, 255), (0, 255, 0))
}

TRACKBAR_WINDOWS = ["image", "image2", "image3"]

def callback(x):
    pass

def setup_trackbars():
    """Setup trackbars untuk penyesuaian HSV range setiap warna."""
    for i, (color_name, (low_range, high_range, _)) in enumerate(COLOR_RANGES.items()):
        window = TRACKBAR_WINDOWS[i]
        cv2.namedWindow(window)
        cv2.createTrackbar(f'lowH_{color_name}', window, low_range[0], 255, callback)
        cv2.createTrackbar(f'highH_{color_name}', window, high_range[0], 255, callback)
        cv2.createTrackbar(f'lowS_{color_name}', window, low_range[1], 255, callback)
        cv2.createTrackbar(f'highS_{color_name}', window, high_range[1], 255, callback)
        cv2.createTrackbar(f'lowV_{color_name}', window, low_range[2], 255, callback)
        cv2.createTrackbar(f'highV_{color_name}', window, high_range[2], 255, callback)

def get_trackbar_values():
    """Mengambil nilai dari trackbars."""
    color_ranges = {}
    for i, (color_name, (_, _, color_bgr)) in enumerate(COLOR_RANGES.items()):
        window = TRACKBAR_WINDOWS[i]
        lowH = cv2.getTrackbarPos(f'lowH_{color_name}', window)
        highH = cv2.getTrackbarPos(f'highH_{color_name}', window)
        lowS = cv2.getTrackbarPos(f'lowS_{color_name}', window)
        highS = cv2.getTrackbarPos(f'highS_{color_name}', window)
        lowV = cv2.getTrackbarPos(f'lowV_{color_name}', window)
        highV = cv2.getTrackbarPos(f'highV_{color_name}', window)
        color_ranges[color_name] = ((np.array([lowH, lowS, lowV]), np.array([highH, highS, highV])), color_bgr)
    return color_ranges

def detect_color(frame, color_range):
    """Deteksi warna tertentu dalam frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, color_range[0], color_range[1])

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=4)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def log_detection(color, timestamp, log_file="detections.txt"):
    """Catat deteksi warna ke file log."""
    with open(log_file, "a") as file:
        file.write(f"{timestamp:.2f}s: {color} detected\n")

def process_frame(frame, color_ranges, timestamp):
    """Proses frame untuk mendeteksi warna dan menampilkan hasilnya."""
    for color_name, (color_range, color_bgr) in color_ranges.items():
        contours = detect_color(frame, color_range)
        for contour in contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
                cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                log_detection(color_name, timestamp)

    return frame

def main(video_path):
    setup_trackbars()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka video.")
        return

    open("detections.txt", "w").close()  # Reset file log

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 640), interpolation=cv2.INTER_AREA)
        color_ranges = get_trackbar_values()
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Waktu dalam detik
        processed_frame = process_frame(frame, color_ranges, timestamp)

        # Tampilkan hasil
        cv2.imshow("Traffic Light Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "dataset/lamp2.mp4"
    main(video_path)
