import cv2
import tensorflow as tf
import numpy as np
import json
import time
import os

from tracking import HandTracker
from drawing import DrawingBoard


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracker = HandTracker()
board = DrawingBoard()


print("Loading AI model...")

model = tf.keras.models.load_model(
    "models/sketch_classifier.keras"
)

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

print("AI model loaded successfully!")
print("Classes:", class_names)



CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

palette_height = 70
color_box_width = 100


DRAW_IDLE_SECONDS = 1.2

last_draw_time = 0.0
sketch_dirty = False

predicted_class = "NONE"
confidence = 0.0


CONFIDENCE_THRESHOLD = 0.80
REAL_IMAGES_DIR = "assets/real_images"
last_shown_label = None


def preprocess_sketch(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    if not np.any(gray):
        return None

    coords = cv2.findNonZero(gray)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = gray[y:y + h, x:x + w]


    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)

    offset_x = (size - w) // 2
    offset_y = (size - h) // 2

    square[
        offset_y:offset_y + h,
        offset_x:offset_x + w
    ] = cropped

    image = cv2.resize(
        square,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype("float32") / 255.0
    image = image.reshape(1, 28, 28, 1)

    return image




def predict_sketch(canvas):
    image = preprocess_sketch(canvas)

    if image is None:
        return "NONE", 0.0

    prediction = model.predict(image, verbose=0)[0]

    index = int(np.argmax(prediction))
    confidence_value = float(prediction[index])
    predicted_label = class_names[index]

    return predicted_label, confidence_value




try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera not found")
            break


        frame = cv2.flip(frame, 1)
        frame = cv2.resize(
            frame,
            (CAMERA_WIDTH, CAMERA_HEIGHT)
        )


        position = tracker.detect(frame)

        if position is not None:
            x, y = position

            # Fingertip cursor.
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            cv2.circle(frame, (x, y), 13, (255, 255, 255), 2)


            if y < palette_height:
                index = x // color_box_width

                if 0 <= index < len(board.colors):
                    name, color = board.colors[index]
                    if name != board.current_color_name:
                        board.set_color(name, color)


                board.previous_point = None

            else:

                board.draw(position)


                last_draw_time = time.monotonic()
                sketch_dirty = True


                predicted_class = "NONE"
                confidence = 0.0

                cv2.putText(
                    frame,
                    "PEN DOWN",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        else:

            board.previous_point = None

            cv2.putText(
                frame,
                "PEN UP",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        
        if (
            sketch_dirty
            and last_draw_time > 0
            and time.monotonic() - last_draw_time >= DRAW_IDLE_SECONDS
        ):
            predicted_class, confidence = predict_sketch(board.canvas)
            sketch_dirty = False

            print(
                f"Prediction: {predicted_class} | "
                f"Confidence: {confidence * 100:.1f}%"
            )


            if predicted_class != "NONE" and confidence >= CONFIDENCE_THRESHOLD:
                label = str(predicted_class).strip().lower()
                image_path = os.path.join(REAL_IMAGES_DIR, f"{label}.jpg")

                if not os.path.exists(image_path):
                    # Also support PNG photos.
                    image_path = os.path.join(REAL_IMAGES_DIR, f"{label}.png")

                if os.path.exists(image_path):
                    real_image = cv2.imread(image_path)
                    if real_image is not None:

                        real_image = cv2.resize(
                            real_image,
                            (500, 400),
                            interpolation=cv2.INTER_AREA
                        )
                        cv2.imshow("AI Real Image", real_image)
                        last_shown_label = label
                        print(f"Showing local real photo: {image_path}")
                    else:
                        print(f"Could not open image file: {image_path}")
                else:
                    print(
                        f"No real photo found for '{label}'. "
                        f"Add {label}.jpg or {label}.png inside "
                        f"{REAL_IMAGES_DIR}/"
                    )
            else:
                last_shown_label = None
                print("Confidence below 80%; real photo will not be shown.")


        board_display = board.canvas.copy()


        for i, (name, color) in enumerate(board.colors):
            x1 = i * color_box_width
            x2 = x1 + color_box_width

            cv2.rectangle(
                board_display,
                (x1, 0),
                (x2, palette_height),
                color,
                -1
            )

            if name == board.current_color_name:
                cv2.rectangle(
                    board_display,
                    (x1, 0),
                    (x2, palette_height),
                    (255, 255, 255),
                    4
                )

            cv2.putText(
                board_display,
                name,
                (x1 + 8, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1
            )

      

        mode = "ERASER" if board.eraser_mode else "DRAW"

        if sketch_dirty:
            prediction_text = "Prediction: WAITING..."
            confidence_text = "Finish drawing"
        elif predicted_class != "NONE":
            prediction_text = f"Prediction: {predicted_class.upper()}"
            confidence_text = f"Confidence: {confidence * 100:.1f}%"
        else:
            prediction_text = "Prediction: --"
            confidence_text = ""

       
        panel_x1, panel_y1 = 10, 285
        panel_x2, panel_y2 = 390, 455

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

        cv2.putText(
            frame,
            f"Mode: {mode}",
            (panel_x1 + 12, panel_y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Color: {board.current_color_name}",
            (panel_x1 + 12, panel_y1 + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            board.current_color,
            2
        )

        cv2.putText(
            frame,
            prediction_text,
            (panel_x1 + 12, panel_y1 + 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        if confidence_text:
            cv2.putText(
                frame,
                confidence_text,
                (panel_x1 + 12, panel_y1 + 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

       
        undo_x1, undo_y1 = 10, 310
        undo_x2, undo_y2 = 130, 350

        cv2.rectangle(
            board_display,
            (undo_x1, undo_y1),
            (undo_x2, undo_y2),
            (70, 70, 70),
            -1
        )

        cv2.rectangle(
            board_display,
            (undo_x1, undo_y1),
            (undo_x2, undo_y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            board_display,
            "UNDO (U)",
            (undo_x1 + 12, undo_y1 + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "CAMERA",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            board_display,
            "AI DRAWING BOARD",
            (390, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        combined = cv2.hconcat([frame, board_display])
        cv2.imshow("AI SketchVision", combined)


        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            board.clear()
            predicted_class = "NONE"
            confidence = 0.0
            sketch_dirty = False
            last_draw_time = 0.0

        elif key == ord("e"):
            board.toggle_eraser()

            predicted_class = "NONE"
            confidence = 0.0

        elif key == ord("u"):
            board.undo()
            predicted_class = "NONE"
            confidence = 0.0
            sketch_dirty = bool(np.any(board.canvas))
            last_draw_time = time.monotonic() if sketch_dirty else 0.0

        elif key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
