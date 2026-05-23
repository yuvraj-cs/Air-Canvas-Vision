import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Initialize Color Arrays for drawing
# Index 0: Blue, 1: Green, 2: Red, 3: Yellow
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_index = 0

# Emoji settings
emoji_enabled = False
emoji_type = 0  # 0: Smiley, 1: Cool, 2: Shocked
emoji_names = ["SMILEY", "COOL", "SHOCKED"]
emoji_toggle_cooldown = 0
emoji_type_cooldown = 0

# Programmatic Emoji drawing functions
def draw_smiley_emoji(frame, x, y, w, h):
    center_x = x + w // 2
    center_y = y + h // 2
    radius = min(w, h) // 2
    if radius <= 0:
        return
    # Face (Yellow Circle)
    cv2.circle(frame, (center_x, center_y), radius, (0, 242, 255), -1)
    cv2.circle(frame, (center_x, center_y), radius, (0, 180, 200), 2)
    # Eyes (Black circles)
    eye_radius = int(radius * 0.1)
    left_eye_center = (center_x - int(radius * 0.3), center_y - int(radius * 0.15))
    right_eye_center = (center_x + int(radius * 0.3), center_y - int(radius * 0.15))
    cv2.circle(frame, left_eye_center, eye_radius, (0, 0, 0), -1)
    cv2.circle(frame, right_eye_center, eye_radius, (0, 0, 0), -1)
    # Mouth (Smile)
    smile_center_y = center_y + int(radius * 0.1)
    cv2.ellipse(frame, (center_x, smile_center_y), (int(radius * 0.45), int(radius * 0.35)), 0, 0, 180, (0, 0, 0), int(radius * 0.08) if radius * 0.08 > 1 else 1)

def draw_cool_emoji(frame, x, y, w, h):
    center_x = x + w // 2
    center_y = y + h // 2
    radius = min(w, h) // 2
    if radius <= 0:
        return
    # Face (Yellow Circle)
    cv2.circle(frame, (center_x, center_y), radius, (0, 242, 255), -1)
    cv2.circle(frame, (center_x, center_y), radius, (0, 180, 200), 2)
    # Sunglasses (Black shapes)
    glass_w = int(radius * 0.5)
    glass_h = int(radius * 0.3)
    left_glass_center_x = center_x - int(radius * 0.35)
    right_glass_center_x = center_x + int(radius * 0.35)
    glass_y = center_y - int(radius * 0.15)
    # Draw left glass
    cv2.rectangle(frame, 
                  (left_glass_center_x - glass_w//2, glass_y - glass_h//2), 
                  (left_glass_center_x + glass_w//2, glass_y + glass_h//2), 
                  (0, 0, 0), -1)
    # Draw right glass
    cv2.rectangle(frame, 
                  (right_glass_center_x - glass_w//2, glass_y - glass_h//2), 
                  (right_glass_center_x + glass_w//2, glass_y + glass_h//2), 
                  (0, 0, 0), -1)
    # Sunglasses bridge
    cv2.line(frame, (left_glass_center_x, glass_y), (right_glass_center_x, glass_y), (0, 0, 0), int(radius * 0.08) if radius * 0.08 > 1 else 1)
    # Mouth (Smile)
    smile_center_y = center_y + int(radius * 0.25)
    cv2.ellipse(frame, (center_x, smile_center_y), (int(radius * 0.45), int(radius * 0.25)), 0, 0, 180, (0, 0, 0), int(radius * 0.08) if radius * 0.08 > 1 else 1)

def draw_shocked_emoji(frame, x, y, w, h):
    center_x = x + w // 2
    center_y = y + h // 2
    radius = min(w, h) // 2
    if radius <= 0:
        return
    # Face (Yellow Circle)
    cv2.circle(frame, (center_x, center_y), radius, (0, 242, 255), -1)
    cv2.circle(frame, (center_x, center_y), radius, (0, 180, 200), 2)
    # Eyes (Wide open circles)
    eye_radius = int(radius * 0.13)
    left_eye_center = (center_x - int(radius * 0.3), center_y - int(radius * 0.15))
    right_eye_center = (center_x + int(radius * 0.3), center_y - int(radius * 0.15))
    cv2.circle(frame, left_eye_center, eye_radius, (255, 255, 255), -1)
    cv2.circle(frame, right_eye_center, eye_radius, (255, 255, 255), -1)
    cv2.circle(frame, left_eye_center, eye_radius, (0, 0, 0), int(radius * 0.03) if radius * 0.03 > 1 else 1)
    cv2.circle(frame, right_eye_center, eye_radius, (0, 0, 0), int(radius * 0.03) if radius * 0.03 > 1 else 1)
    cv2.circle(frame, left_eye_center, int(eye_radius * 0.5), (0, 0, 0), -1)
    cv2.circle(frame, right_eye_center, int(eye_radius * 0.5), (0, 0, 0), -1)
    # Mouth (Perfect O shape)
    mouth_center_y = center_y + int(radius * 0.35)
    cv2.circle(frame, (center_x, mouth_center_y), int(radius * 0.18), (0, 0, 0), -1)

def draw_face_emoji(frame, bbox, type_index):
    x, y, w, h = bbox
    if type_index == 0:
        draw_smiley_emoji(frame, x, y, w, h)
    elif type_index == 1:
        draw_cool_emoji(frame, x, y, w, h)
    elif type_index == 2:
        draw_shocked_emoji(frame, x, y, w, h)

# Deque-like structures to store points of different colors
bpoints = [[]]
gpoints = [[]]
rpoints = [[]]
ypoints = [[]]

# Setup the UI Window
canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255

# Open Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Cooldown decrements
    if emoji_toggle_cooldown > 0:
        emoji_toggle_cooldown -= 1
    if emoji_type_cooldown > 0:
        emoji_type_cooldown -= 1

    # Draw UI Buttons on the webcam frame
    cv2.rectangle(frame, (40, 10), (140, 65), (0, 0, 0), -1)
    cv2.putText(frame, "CLEAR", (49, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    for i, color in enumerate(colors):
        cv2.rectangle(frame, (200 + i*130, 10), (300 + i*130, 65), color, -1)
    
    cv2.putText(frame, "BLUE", (225, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (350, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (485, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (605, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Emoji Toggle Button (Green if enabled, Grey if disabled)
    emoji_toggle_color = (0, 200, 0) if emoji_enabled else (128, 128, 128)
    cv2.rectangle(frame, (720, 10), (850, 65), emoji_toggle_color, -1)
    emoji_toggle_text = "EMOJI ON" if emoji_enabled else "EMOJI OFF"
    cv2.putText(frame, emoji_toggle_text, (730, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Emoji Type Button (Active only if emoji is enabled)
    emoji_type_color = (0, 242, 255) if emoji_enabled else (80, 80, 80)
    cv2.rectangle(frame, (880, 10), (1030, 65), emoji_type_color, -1)
    cv2.putText(frame, emoji_names[emoji_type], (895, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if emoji_enabled else (180, 180, 180), 2, cv2.LINE_AA)

    # Process face detection if emoji mode is enabled
    face_bboxes = []
    if emoji_enabled:
        face_result = face_detector.process(rgb_frame)
        if face_result.detections:
            for detection in face_result.detections:
                bbox_data = detection.location_data.relative_bounding_box
                fx = int(bbox_data.xmin * w)
                fy = int(bbox_data.ymin * h)
                fw = int(bbox_data.width * w)
                fh = int(bbox_data.height * h)
                
                # Pad/expand bounding box to nicely cover the head/face
                padding_w = int(fw * 0.1)
                padding_h = int(fh * 0.15)
                fx_padded = max(0, fx - padding_w)
                fy_padded = max(0, fy - padding_h)
                fw_padded = min(w - fx_padded, fw + 2 * padding_w)
                fh_padded = min(h - fy_padded, fh + 2 * padding_h)
                
                face_bboxes.append((fx_padded, fy_padded, fw_padded, fh_padded))

    # Process hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the hand for visual feedback
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get Coordinates of Index finger tip (Landmark 8) and Thumb tip (Landmark 4)
            idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Calculate distance between thumb and index finger to detect "clicking" / "drawing mode"
            distance = np.sqrt((cx - tx)**2 + (cy - ty)**2)

            # Draw a circle on the index tip
            cv2.circle(frame, (cx, cy), 10, colors[color_index], -1)

            # Check if finger is over the top button layout area
            if cy <= 65:
                if 40 <= cx <= 140: # Clear Button
                    bpoints = [[]]
                    gpoints = [[]]
                    rpoints = [[]]
                    ypoints = [[]]
                    canvas[:, :, :] = 255
                elif 200 <= cx <= 300:
                    color_index = 0 # Blue
                elif 330 <= cx <= 430:
                    color_index = 1 # Green
                elif 460 <= cx <= 560:
                    color_index = 2 # Red
                elif 590 <= cx <= 690:
                    color_index = 3 # Yellow
                elif 720 <= cx <= 850:
                    if emoji_toggle_cooldown == 0:
                        emoji_enabled = not emoji_enabled
                        emoji_toggle_cooldown = 15  # Cooldown of 15 frames
                elif 880 <= cx <= 1030:
                    if emoji_type_cooldown == 0 and emoji_enabled:
                        emoji_type = (emoji_type + 1) % 3
                        emoji_type_cooldown = 15  # Cooldown of 15 frames
            else:
                # If distance is small, treat it as "Finger up/drawing deactivated" or "Hovering"
                # Adjust this threshold based on your camera's depth/resolution
                if distance > 40:
                    if color_index == 0:
                        bpoints[-1].append((cx, cy))
                    elif color_index == 1:
                        gpoints[-1].append((cx, cy))
                    elif color_index == 2:
                        rpoints[-1].append((cx, cy))
                    elif color_index == 3:
                        ypoints[-1].append((cx, cy))
                else:
                    # Append empty list to create a break in drawing lines when fingers pinch together
                    for pts in [bpoints, gpoints, rpoints, ypoints]:
                        if len(pts[-1]) > 0:
                            pts.append([])

    else:
        # If no hand detected, add empty arrays to break the lines seamlessly
        for pts in [bpoints, gpoints, rpoints, ypoints]:
            if len(pts[-1]) > 0:
                pts.append([])

    # Draw lines on both frames
    points_list = [bpoints, gpoints, rpoints, ypoints]
    for i, stroke_group in enumerate(points_list):
        for stroke in stroke_group:
            for k in range(1, len(stroke)):
                if stroke[k - 1] is None or stroke[k] is None:
                    continue
                cv2.line(frame, stroke[k - 1], stroke[k], colors[i], 3)
                cv2.line(canvas, stroke[k - 1], stroke[k], colors[i], 3)

    # Draw emojis on detected faces
    for bbox in face_bboxes:
        draw_face_emoji(frame, bbox, emoji_type)

    # Display Windows
    cv2.imshow("Air Canvas - Camera Feed", frame)
    cv2.imshow("Air Canvas - Whiteboard", canvas)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()