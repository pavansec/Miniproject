from flask import Flask, render_template, Response
import cv2
import os
import pickle
import numpy as np
from gtts import gTTS
import pygame
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

app = Flask(__name__)

# Load the pre-trained model and scaler
model_dict = pickle.load(open('./model/model.p', 'rb'))
model = model_dict['model']
scaler = model_dict['scaler']  # Load the scaler

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize the labels dictionary (Tamil characters)
labels_dict = {
    0: 'அ', 1: 'வாங்க', 2: 'வணக்கம்', 3: 'ஜனார்த்தனன்', 4: 'எப்புடி இருக்கேங்க', 
    5: 'ஊ', 6: 'எ', 7: 'ஏ', 8: 'ஐ', 9: 'ஒ', 10: 'ஓ', 11: 'ஔ',
}

# Load a Tamil font using PIL
tamil_font = ImageFont.truetype("NotoSansTamil-VariableFont_wdth,wght.ttf", 40)

# Initialize pygame for audio playback
pygame.mixer.init()

# Create a folder for audio files
audio_dir = 'audio_files'
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# Generate and save audio for each Tamil character
for key, tamil_character in labels_dict.items():
    audio_file = os.path.join(audio_dir, f'{key}.mp3')
    if not os.path.exists(audio_file):
        tts = gTTS(text=tamil_character, lang='ta')
        tts.save(audio_file)

# Minimum confidence threshold to display predictions
CONFIDENCE_THRESHOLD = 0.75

def gen_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        # Always try to capture the frame from the camera
        success, frame = cap.read()
        if not success:
            continue  # If we fail to get a frame, continue to the next iteration

        try:
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                        data_aux.append(hand_landmarks.landmark[i].y - min(y_))
                        
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Scale the input data using the scaler
                data_scaled = scaler.transform([np.asarray(data_aux)])

                # Get the prediction probabilities
                prediction_probs = model.predict_proba(data_scaled)[0]
                predicted_value = np.argmax(prediction_probs)

                # Only display if the confidence exceeds the threshold
                if prediction_probs[predicted_value] >= CONFIDENCE_THRESHOLD and predicted_value in labels_dict:
                    predicted_character = labels_dict[predicted_value]

                    # Use PIL to render Tamil characters on the frame
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(frame_pil)
                    draw.text((x1, y1 - 50), predicted_character, font=tamil_font, fill=(57, 255, 20))
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                    # Play the corresponding audio
                    audio_file = os.path.join(audio_dir, f'{predicted_value}.mp3')
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.load(audio_file)
                        pygame.mixer.music.play()

                else:
                    print("Low confidence or unknown gesture. No output.")

        except Exception as e:
            # Log the error, but do not interrupt the frame generation
            print(f"Error during processing: {str(e)}")
            # The video feed will still display, just without annotations

        # Encode the frame and send it to the client
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
