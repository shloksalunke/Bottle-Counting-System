from ultralytics import YOLO
import cv2
import time
from datetime import datetime

# ==========================
# LOAD MODEL
# ==========================
model = YOLO("best.pt")

# ==========================
# VIDEO
# ==========================
cap = cv2.VideoCapture("Generate_a_realistic_second.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = int(cap.get(cv2.CAP_PROP_FPS))

# ==========================
# SAVE OUTPUT VIDEO
# ==========================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "Bottle_Counting_Output.mp4",
    fourcc,
    fps_video,
    (frame_width, frame_height)
)

# ==========================
# COUNTING LINE
# ==========================
LINE_X = 1000

counted_ids = set()
bottle_count = 0

prev_time = time.time()

# ==========================
# LOOP
# ==========================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------
    # YOLO Tracking
    # --------------------------
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        verbose=False
    )

    # --------------------------
    # FPS
    # --------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # --------------------------
    # Dashboard Overlay
    # --------------------------
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (500, 190),
        (20, 20, 20),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0
    )

    # --------------------------
    # Dashboard Text
    # --------------------------
    cv2.putText(
        frame,
        "AI Bottle Counting System",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Bottles : {bottle_count}",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Status : RUNNING",
        (25, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # --------------------------
    # Timestamp
    # --------------------------
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    cv2.putText(
        frame,
        timestamp,
        (frame_width - 320, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------
    # Counting Line
    # --------------------------
    cv2.line(
        frame,
        (LINE_X, 0),
        (LINE_X, frame_height),
        (0, 165, 255),
        4
    )

    cv2.putText(
        frame,
        "COUNT LINE",
        (LINE_X - 140, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2
    )

    # --------------------------
    # Tracking Results
    # --------------------------
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, ids, confs):

            x1, y1, x2, y2 = map(int, box)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Professional Box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            # Center Point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            # Label
            label = f"ID:{track_id}  Conf:{conf:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

            # --------------------------
            # Counting Logic
            # Bottles moving LEFT -> RIGHT
            # --------------------------
            if center_x > LINE_X and track_id not in counted_ids:

                bottle_count += 1
                counted_ids.add(track_id)

                print(f"Bottle Counted: {bottle_count}")

    # --------------------------
    # Bottom Status Bar
    # --------------------------
    cv2.rectangle(
        frame,
        (0, frame_height - 40),
        (frame_width, frame_height),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        frame,
        "YOLOv8 + ByteTrack | Real-Time Bottle Counting",
        (20, frame_height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # --------------------------
    # Show Frame
    # --------------------------
    cv2.imshow("Bottle Counter Pro", frame)

    # Save Output
    out.write(frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

# ==========================
# RELEASE
# ==========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("=" * 40)
print("PROCESS COMPLETED")
print(f"TOTAL BOTTLES COUNTED : {bottle_count}")
print("OUTPUT SAVED : Bottle_Counting_Output.mp4")
print("=" * 40)