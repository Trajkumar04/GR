import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

def get_hand_landmarks(image):
    print("Processing image for hand landmarks...")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        print("Hand landmarks detected.")
        return results.multi_hand_landmarks[0]
    print("No hand landmarks detected.")
    return None

