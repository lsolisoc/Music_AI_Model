import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def count_extended_fingers(hand_landmarks, image_width, image_height):
    # Only count index, middle, ring, pinky (ignore thumb for stability)
    tip_indices = [8, 12, 16, 20]
    pip_indices = [6, 10, 14, 18]

    extended = 0
    for tip, pip in zip(tip_indices, pip_indices):
        tip_y = hand_landmarks.landmark[tip].y * image_height
        pip_y = hand_landmarks.landmark[pip].y * image_height
        if tip_y < pip_y:  # fingertip above pip → extended
            extended += 1

    return extended


def get_hand_center(hand_landmarks):
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    
    # normalized 0–1
    return sum(xs) / len(xs), sum(ys) / len(ys)


def get_hand_openness(hand_landmarks):
    # Wrist index is 0
    wrist = hand_landmarks.landmark[0]
    # Fingertip indices: thumb, index, middle, ring, pinky
    tip_indices = [4, 8, 12, 16, 20]

    dists = []
    for idx in tip_indices:
        tip = hand_landmarks.landmark[idx]
        dx = tip.x - wrist.x
        dy = tip.y - wrist.y
        dist = (dx**2 + dy**2) ** 0.5
        dists.append(dist)

    # Average normalized distance
    return sum(dists) / len(dists)


def normalize(value, min_val, max_val):
    # Setting the scale from 0–1
    value = max(min_val, min(max_val, value))
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0

# Store previous positions to calculate velocity
prev_centers = {}

def compute_velocity(label, cx, cy):
    # Very simple frame-to-frame velocity based on center movement
    if label not in prev_centers:
        prev_centers[label] = (cx, cy)
        return 0.0

    prev_x, prev_y = prev_centers[label]
    dx = cx - prev_x
    dy = cy - prev_y
    prev_centers[label] = (cx, cy)

    # Euclidean distance per frame to measure speed of movement
    return (dx**2 + dy**2) ** 0.5


def main():
    cap = cv2.VideoCapture(0)

    # window setup
    cv2.namedWindow("Hand Tracking - Press q to quit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracking - Press q to quit", 800, 600)

    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                image_height, image_width = frame.shape[:2]

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label  # 'Left' or 'Right'

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Left hand → finger count (for mood and creativity)
                    if label == "Left":
                        fingers = count_extended_fingers(
                            hand_landmarks, image_width, image_height
                        )
                        cv2.putText(
                            frame,
                            f"Left fingers: {fingers}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                        print(f"LEFT HAND - fingers extended: {fingers}")

                    # Right hand: X/Y center + openness (for pitch/rhythm/intensity)
                    elif label == "Right":
                        cx, cy = get_hand_center(hand_landmarks)   # normalized 0–1
                        raw_open = get_hand_openness(hand_landmarks)
                        open_norm = normalize(raw_open, 0.20, 0.50)

                        vel = compute_velocity(label, cx, cy)
                        vel_norm = normalize(vel, 0.0, 0.05)  # tune 0.05 depending speed of movement

                        cv2.putText(
                            frame,
                            f"Right X:{cx:.2f} Y:{cy:.2f} Open:{open_norm:.2f} Vel:{vel_norm:.2f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                        )
                        print(
                            f"RIGHT HAND - center: X={cx:.2f}, Y={cy:.2f}, "
                            f"open_raw={raw_open:.2f}, open_norm={open_norm:.2f}, vel_norm={vel_norm:.2f}"
                        )


            cv2.imshow("Hand Tracking - Press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
