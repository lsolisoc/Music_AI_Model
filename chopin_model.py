import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import mido
from mido import Message
import time


class LSTMMusicModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# === MODEL LOADING ===
MODEL_PATH = "models/chopin_lstm.pt"
SEQ_LEN = 48

save_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

model = LSTMMusicModel(save_dict["vocab_size"]).to(DEVICE)
model.load_state_dict(save_dict["model_state"])
model.eval()

token_to_idx = save_dict["token_to_idx"]
idx_to_token = save_dict["idx_to_token"]
VOCAB_SIZE = save_dict["vocab_size"]

print(f"Loaded model with vocab size {VOCAB_SIZE}")

# === MIDI OUTPUT ===
outport = mido.open_output()
print("MIDI output ready")

# Rolling context for model
context = torch.zeros(1, SEQ_LEN, dtype=torch.long).to(DEVICE)
context.fill_(list(token_to_idx.values())[0])  # start with first token

# Gesture state
left_fingers = 0          # temperature / creativity
right_x = 0.5             # rhythm: 0=slow, 1=fast
right_y = 0.5             # pitch: 0=high, 1=low
right_openness = 0.0      # layers + loudness
last_note_time = 0.0
NOTE_INTERVAL = 0.25      # base seconds between notes

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)


def count_fingers(landmarks, handedness):
    """Return 0–4 finger count; treat closed fist as 0 (thumb ignored)."""
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]

    count = 0
    for tip, pip in zip(tip_ids, pip_ids):
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
            count += 1
    return count if count > 0 else 0


def hand_openness(landmarks):
    """0=closed fist, 1=open hand (based on thumb–index distance)."""
    thumb_tip = landmarks.landmark[4].x
    index_tip = landmarks.landmark[8].x
    dist = abs(thumb_tip - index_tip)
    return min(dist * 5, 1.0)


cap = cv2.VideoCapture(0)

print("Starting gesture-controlled Chopin generation...")
print("Left hand: fingers = style (temperature)")
print("Right hand: X=speed, Y=pitch, openness=layers/loudness")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror camera so movements look natural
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_type = handedness.classification[0].label

            # Palm center normalized 0–1
            cx = hand_landmarks.landmark[9].x
            cy = hand_landmarks.landmark[9].y

            fingers = count_fingers(hand_landmarks, handedness)
            openness = hand_openness(hand_landmarks)

            # Use RIGHT physical hand for music control
            if hand_type == "Right":
                right_x = cx          # 0 (left) → 1 (right) : speed
                right_y = cy          # 0 (top) → 1 (bottom): pitch
                right_openness = openness
            else:
                left_fingers = fingers

    # --- CLEAR STATUS TEXT AT TOP ---
    status = (
        f"Left fingers (temp): {left_fingers}  |  "
        f"Right X(speed): {right_x:.2f}  Y(pitch): {right_y:.2f}  "
        f"Openness: {right_openness:.2f}"
    )
    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # === MUSIC GENERATION ===
    now = time.time()

    # Right X controls rhythm speed: mapped so right→faster smoothly, floor at 0.15
    speed_factor = max(0.15, 1.0 - right_x)

    if now - last_note_time > NOTE_INTERVAL * speed_factor:

        with torch.no_grad():
            logits, _ = model(context)
            probs = torch.softmax(logits[0, -1], dim=0)

            # Left hand controls temperature (creativity)
            if left_fingers == 0:
                temp = 0.65   # closer to training data to reduce randomness
            elif left_fingers == 1:
                temp = 0.75
            elif left_fingers == 2:
                temp = 0.85
            else:
                temp = 0.95  # mildly exploratory

            probs = probs ** (1.0 / temp)
            probs = probs / probs.sum()

            next_idx = torch.multinomial(probs, 1).item()
            next_pitch = idx_to_token[next_idx]

            if next_pitch != -1:
                # Clamp model pitch output mid-keyboard range
                base_pitch = int(next_pitch)
                base_pitch = max(48, min(84, base_pitch))  # C3–C6

                # Y controls pitch register shift: tighter range ±5 semitones
                pitch_shift = int((0.5 - right_y) * 10)  # -5 to +5 semitones
                final_pitch = max(43, min(88, base_pitch + pitch_shift))  # Slightly wider safe range

                # Openness determines chord layers: 1, 2, or 5 notes for richer chords
                if right_openness < 0.3:
                    layer_count = 1
                elif right_openness < 0.6:
                    layer_count = 2
                else:
                    layer_count = 5

                # Chopin-style chord intervals: root, 3rd, 5th, 7th, 9th
                offsets = [0, 4, 7, 10, 14]

                velocity = int(50 + right_openness * 75)  # velocity 50–125

                # Play chord notes with small intervals for musical effect
                for i in range(layer_count):
                    offset = offsets[i]
                    layer_pitch = max(36, min(96, final_pitch + offset))
                    outport.send(
                        Message("note_on", note=layer_pitch, velocity=velocity, time=0)
                    )
                    time.sleep(0.015)  # slightly shorter delay for smoother chord

                last_note_time = now

                # Update LSTM context with chosen token
                context = torch.roll(context, -1, dims=1)
                context[0, -1] = next_idx

    cv2.imshow("Chopin Gesture AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
outport.close()
