import cv2
import numpy as np
import tensorflow as tf
import threading
from playsound import playsound
import os


from hand_tracking import get_hand_landmarks
from effects import (
    display_not_effect, display_heart_effect, display_yes_effect,
    display_rock_effect, display_shaka_effect, display_ok_effect,
    display_peace_effect, display_stop_effect
)

# Load the TFLite model and allocate tensors.
print("Reached video_capture.py...")
interpreter = tf.lite.Interpreter(model_path="../models/gesture_recognition_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Flag to keep track of the current gesture and sound thread
current_gesture = None
sound_thread = None

# Counter for maintaining the same gesture over frames
gesture_counter = 0
consistent_gesture_threshold = 10

# This is a function that recognizes the gesture in the input image
def recognize_gesture(image):
    print("Recognizing gesture...")
    print("Input image shape:", image.shape)

    image = cv2.resize(image, (224, 224))
    print("Resized image shape:", image.shape)

    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("Gesture recognition result:", output_data)

    return np.argmax(output_data)

# This is a function that plays the desired sound effect once
def play_sound_once(sound_file):
    global sound_thread
    if sound_thread is None or not sound_thread.is_alive():
        sound_thread = threading.Thread(target=playsound, args=(sound_file,), daemon=True)
        sound_thread.start()

# This is the main function that captures the video feed and processes it
def main():
    global current_gesture, sound_thread, gesture_counter

    print("Starting video capture...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        hand_landmarks = get_hand_landmarks(frame)
        if hand_landmarks:
            gesture_id = recognize_gesture(frame)
            print(f"Gesture ID: {gesture_id}")

            if gesture_id == current_gesture:
                gesture_counter += 1
            else:
                gesture_counter = 0
                current_gesture = gesture_id

            if gesture_counter >= consistent_gesture_threshold:
                # Plays the sound if it hasn't played yet
                if gesture_counter == consistent_gesture_threshold:
                    if gesture_id == 0:
                        play_sound_once('../sounds/no_sound.mp3')
                    elif gesture_id == 1:
                        play_sound_once('../sounds/heart_sound.mp3')
                    elif gesture_id == 2:
                        play_sound_once('../sounds/ok_sound.mp3')
                    elif gesture_id == 3:
                        play_sound_once('../sounds/peace_sound.mp3')
                    elif gesture_id == 4:
                        play_sound_once('../sounds/rock_sound.mp3')
                    elif gesture_id == 5:
                        play_sound_once('../sounds/shaka_sound.mp3')
                    elif gesture_id == 6:
                        play_sound_once('../sounds/stop_sound.mp3')

                # Display the effect
                if gesture_id == 0:
                    frame = display_not_effect(frame)
                elif gesture_id == 1:
                    frame = display_heart_effect(frame)
                elif gesture_id == 2:
                    frame = display_ok_effect(frame)
                elif gesture_id == 3:
                    frame = display_peace_effect(frame)
                elif gesture_id == 4:
                    frame = display_rock_effect(frame)
                elif gesture_id == 5:
                    frame = display_shaka_effect(frame)
                elif gesture_id == 6:
                    frame = display_stop_effect(frame)
                elif gesture_id == 7:
                    frame = display_yes_effect(frame)

            cv2.imshow('frame', frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()