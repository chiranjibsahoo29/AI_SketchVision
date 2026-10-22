import cv2
import mediapipe as mp


class HandTracker:

    def __init__(self):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )


        self.smooth_x = None
        self.smooth_y = None
        self.smoothing_factor = 0.35


        self.gesture_frames = 0
        self.required_frames = 2

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:

            self.reset_tracking()
            return None

        hand = results.multi_hand_landmarks[0]
        landmarks = hand.landmark

        index_tip = landmarks[8]
        index_pip = landmarks[6]

        middle_tip = landmarks[12]
        middle_pip = landmarks[10]

        ring_tip = landmarks[16]
        ring_pip = landmarks[14]

        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]

        h, w, _ = frame.shape


        index_up = index_tip.y < index_pip.y

        middle_up = middle_tip.y < middle_pip.y
        ring_up = ring_tip.y < ring_pip.y
        pinky_up = pinky_tip.y < pinky_pip.y

        drawing_mode = (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        )

        if not drawing_mode:

            self.reset_tracking()
            return None


        self.gesture_frames += 1

        if self.gesture_frames < self.required_frames:
            return None


        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        # Initialize smoothing
        if self.smooth_x is None:

            self.smooth_x = x
            self.smooth_y = y

        else:

            self.smooth_x = int(
                self.smoothing_factor * x
                + (1 - self.smoothing_factor)
                * self.smooth_x
            )

            self.smooth_y = int(
                self.smoothing_factor * y
                + (1 - self.smoothing_factor)
                * self.smooth_y
            )

        return self.smooth_x, self.smooth_y

    def reset_tracking(self):

        self.smooth_x = None
        self.smooth_y = None
        self.gesture_frames = 0