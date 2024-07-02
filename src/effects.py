import cv2
import numpy as np
from playsound import playsound
import threading


def verify_alpha_channel(frame):
    try:
        frame.shape[3]  # looking for the alpha channel
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame

# This function displays the 'not' effect when the thumbs down hand gesture is recognized
def display_not_effect(frame):
    print("Displaying invert effect...")

    # Changing color of the overlay
    frame = cv2.bitwise_not(frame)

    # Code for adding the 'not' image
    # Adding the 'not' image
    not_image = cv2.imread('../data/effects/not_effect.png')
    not_size = 50
    not_image = cv2.resize(not_image, (not_size, not_size))

    # Create a rotated version of the 'not' image
    center = (not_size // 2, not_size // 2)
    angle = 15  # Rotation angle in degrees
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_not_image = cv2.warpAffine(not_image, M, (not_size, not_size))

    img2gray = cv2.cvtColor(rotated_not_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(mask)

    # Frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Region of interest (ROI) coordinates
    positions = [
        (10, 10),  # Top-left
        (frame_width - not_size - 10, 10),  # Top-right
        (10, frame_height - not_size - 10),  # Bottom-left
        (frame_width - not_size - 10, frame_height - not_size - 10)  # Bottom-right
    ]

    for (x, y) in positions:
        # Define the region of interest
        roi = frame[y:y + not_size, x:x + not_size]
        # Black-out the area of the 'not' image in the ROI
        roi_bg = cv2.bitwise_and(roi, roi, mask = inverted_mask)
        # Take only region of the 'not' image
        roi_fg = cv2.bitwise_and(rotated_not_image, rotated_not_image, mask=mask)
        # Place the 'not' image in the frame
        dst = cv2.add(roi_bg, roi_fg)
        frame[y:y + not_size, x:x + not_size] = dst
 
    return frame

# This function displays the heart effect when the heart hand gesture is recognized
def display_heart_effect(frame, intensity=0.5):
    print("Displaying heart effect...")
    frame = verify_alpha_channel(frame)
    frame_height, frame_width, frame_c = frame.shape

    # Adding the heart image
    heart_image = cv2.imread('../data/effects/heart_effect.png', cv2.IMREAD_UNCHANGED)
    
    if heart_image is None:
        print("Error: Heart effect image not found.")
        return frame

    # Resize the heart image to cover the entire frame
    heart_image = cv2.resize(heart_image, (frame_width, frame_height))

    # Ensure the heart image has an alpha channel
    if heart_image.shape[2] == 3:
        heart_image = cv2.cvtColor(heart_image, cv2.COLOR_BGR2BGRA)
    
    # Create masks for blending
    heart_alpha = heart_image[:, :, 3] / 255.0
    frame_alpha = 1.0 - heart_alpha

    # Blend the heart image with the frame
    for c in range(0, 3):
        frame[:, :, c] = (heart_alpha * heart_image[:, :, c] +
                          frame_alpha * frame[:, :, c])

    # Changing color of the overlay
    blue = 0
    green = 0
    red = 255
    heart = (blue, green, red, 1)
    overlay = np.full((frame_height, frame_width, 4), heart, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


# This function displays the 'yes' effect when the thumbs up hand gesture is recognized
def display_yes_effect(frame, intensity=0.5):
    print("Displaying yes effect...")

    # Changing color of the overlay
    frame = verify_alpha_channel(frame)
    frame_height, frame_width, frame_c = frame.shape
    blue = 0
    green = 255
    red = 0
    yes = (blue, green, red, 1)
    overlay = np.full((frame_height, frame_width, 4), yes, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Adding the yes image
    yes_image = cv2.imread('../data/effects/yes_effect.png')
    yes_size = 100
    yes_image = cv2.resize(yes_image, (yes_size, yes_size))

    # Create a rotated version of the 'yes' image
    center = (yes_size // 2, yes_size // 2)
    angle = 15  # Rotation angle in degrees
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_yes_image = cv2.warpAffine(yes_image, M, (yes_size, yes_size))

    img2gray = cv2.cvtColor(rotated_yes_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(mask)

    # Frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Region of interest (ROI) coordinates
    positions = [
        (10, 10),  # Top-left
        (frame_width - yes_size - 10, 10),  # Top-right
        (10, frame_height - yes_size - 10),  # Bottom-left
        (frame_width - yes_size - 10, frame_height - yes_size - 10)  # Bottom-right
    ]

    for (x, y) in positions:
        # Define the region of interest
        roi = frame[y:y + yes_size, x:x + yes_size]
        # Black-out the area of the 'yes' image in the ROI
        roi_bg = cv2.bitwise_and(roi, roi, mask = inverted_mask)
        # Take only region of the 'yes' image
        roi_fg = cv2.bitwise_and(rotated_yes_image, rotated_yes_image, mask=mask)
        # Place the 'yes' image in the frame
        dst = cv2.add(roi_bg, roi_fg)
        frame[y:y + yes_size, x:x + yes_size] = dst

    return frame

# This function displays the 'rock' effect when the rock hand gesture is recognized
def display_rock_effect(frame, intensity=0.5):
    print("Displaying rock effect...")

    # Changing color of the overlay
    frame = verify_alpha_channel(frame)
    frame_height, frame_width, frame_c = frame.shape


    # Adding the rock image
    rock_image = cv2.imread('../data/effects/rock_effect.png', cv2.IMREAD_UNCHANGED)
    
    if rock_image is None:
        print("Error: Rock effect image not found.")
        return frame

    # Resize the rock image to cover the entire frame
    rock_image = cv2.resize(rock_image, (frame_width, frame_height))

    # Ensure the rock image has an alpha channel
    if rock_image.shape[2] == 3:
        rock_image = cv2.cvtColor(rock_image, cv2.COLOR_BGR2BGRA)
    
    # Create masks for blending
    rock_alpha = rock_image[:, :, 3] / 255.0
    frame_alpha = 1.0 - rock_alpha

    # Blend the rock image with the frame
    for c in range(0, 3):
        frame[:, :, c] = (rock_alpha * rock_image[:, :, c] +
                          frame_alpha * frame[:, :, c])

    # Changing color of the overlay
    blue = 255
    green = 0
    red = 0
    rock = (blue, green, red, 1)
    overlay = np.full((frame_height, frame_width, 4), rock, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


# This function displays the 'shaka' effect when the shaka hand gesture is recognized
def display_shaka_effect(frame, intensity=0.5):
    print("Displaying shaka effect...")
    frame = verify_alpha_channel(frame)
    frame_height, frame_width, frame_c = frame.shape

    blue = 146
    green = 208
    red = 255
    shaka = (blue, green, red, 1)
    overlay = np.full((frame_height, frame_width, 4), shaka, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Adding the shaka image
    shaka_image = cv2.imread('../data/effects/new_shaka_effect.png', cv2.IMREAD_UNCHANGED)
    
    if shaka_image is None:
        print("Error: Shaka effect image not found.")
        return frame

    # Resize the shaka image to cover the entire frame
    shaka_image = cv2.resize(shaka_image, (frame_width, frame_height))

    # Ensure the shaka image has an alpha channel
    if shaka_image.shape[2] == 3:
        shaka_image = cv2.cvtColor(shaka_image, cv2.COLOR_BGR2BGRA)
    
    # Create masks for blending
    shaka_alpha = shaka_image[:, :, 3] / 255.0
    frame_alpha = 1.0 - shaka_alpha

    # Blend the shaka image with the frame
    for c in range(0, 3):
        frame[:, :, c] = (shaka_alpha * shaka_image[:, :, c] +
                          frame_alpha * frame[:, :, c])

    
    return frame

def display_ok_effect(frame, intensity=0.5):
    return frame

# This function displays the 'peace' effect when the peace hand gesture is recognized
def display_peace_effect(frame, intensity=0.5):

    frame = verify_alpha_channel(frame)
    frame_height, frame_width, frame_c = frame.shape

    # Adding the peace image
    peace_image = cv2.imread('../data/effects/peace_effect.png', cv2.IMREAD_UNCHANGED)

    if peace_image is None:
        print("Error: Peace effect image not found.")
        return frame
    
    # Resize the peace image to cover the entire frame
    peace_image = cv2.resize(peace_image, (frame_width, frame_height))

    # Ensure the peace image has an alpha channel
    if peace_image.shape[2] == 3:
        peace_image = cv2.cvtColor(peace_image, cv2.COLOR_BGR2BGRA)

    # Create masks for blending
    peace_alpha = peace_image[:, :, 3] / 255.0
    frame_alpha = 1.0 - peace_alpha

    # Blend the peace image with the frame
    for c in range(0, 3):
        frame[:, :, c] = (peace_alpha * peace_image[:, :, c] +
                          frame_alpha * frame[:, :, c])
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame

def display_stop_effect(frame, intensity=0.5):
    print("Displaying stop effect...")
    frame = verify_alpha_channel(frame)
    frame_height, frame_width, frame_c = frame.shape
    blue = 0
    green = 0
    red = 160
    stop = (blue, green, red, 1)
    overlay = np.full((frame_height, frame_width, 4), stop, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


    # Adding the stop image
    stop_image = cv2.imread('../data/effects/stop_effect.png')
    stop_size = 100
    stop_image = cv2.resize(stop_image, (stop_size, stop_size))

    # Create a rotated version of the 'not' image
    center = (stop_size // 2, stop_size // 2)
    angle = 15  # Rotation angle in degrees
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_stop_image = cv2.warpAffine(stop_image, M, (stop_size, stop_size))

    img2gray = cv2.cvtColor(rotated_stop_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(mask)

    # Frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Region of interest (ROI) coordinates
    positions = [
        (10, 10),  # Top-left
        (frame_width - stop_size - 10, 10),  # Top-right
        (10, frame_height - stop_size - 10),  # Bottom-left
        (frame_width - stop_size - 10, frame_height - stop_size - 10)  # Bottom-right
    ]

    for (x, y) in positions:
        # Define the region of interest
        roi = frame[y:y + stop_size, x:x + stop_size]
        
        # Black-out the area of the 'not' image in the ROI
        roi_bg = cv2.bitwise_and(roi, roi, mask = inverted_mask)
        
        # Take only region of the 'not' image
        roi_fg = cv2.bitwise_and(rotated_stop_image, rotated_stop_image, mask=mask)
        
        # Place the 'not' image in the frame
        dst = cv2.add(roi_bg, roi_fg)
        frame[y:y + stop_size, x:x + stop_size] = dst

    return frame


