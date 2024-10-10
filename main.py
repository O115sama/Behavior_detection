import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import csv
from datetime import datetime
from emailnotify import send_email

model = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

actions = [
    'punching person',
    'punching person (boxing)',
    'smoking hookah',
    'smoking',
    'slapping',
    'fighting',
    'headbutting',
    'wrestling',
    'side kick',
    'drop kicking'
]

frame_buffer = []
BUFFER_SIZE = 20  
alert_duration = 150  
alert_counter = 0
alert_active = False  

if not os.path.exists('photo'):
    os.makedirs('photo')


csv_file_path = 'events.csv'
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Detected Action'])  

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return frame

def detect_behavior(frames):
    input_frames = np.stack(frames, axis=0)
    input_frames = np.expand_dims(input_frames, axis=0)

    model_input = tf.convert_to_tensor(input_frames, dtype=tf.float32)
    outputs = model(rgb_input=model_input)

    logits = outputs['default']
    probabilities = tf.nn.softmax(logits)

    max_index = tf.argmax(probabilities, axis=-1).numpy()[0]

    if max_index >= len(actions):
        detected_action = 'none'
    else:
        detected_action = actions[max_index]

    if tf.reduce_max(probabilities).numpy() < 0.2:  
        detected_action = 'none'

    return detected_action

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

prev_frame = None

while True:
    ret, frame = cap.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray_frame)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            if np.sum(thresh) > 15000:
                processed_frame = preprocess_frame(frame)
                frame_buffer.append(processed_frame)

                if len(frame_buffer) == BUFFER_SIZE:
                    detected_action = detect_behavior(frame_buffer)
                    frame_buffer = []

                    if detected_action in ['punching person', 'punching person (boxing)', 'smoking hookah', 'smoking', 'slapping', 'fighting', 'headbutting', 'wrestling', 'side kick', 'drop kicking']:
                        alert_counter += 1
                        alert_active = True
                        out.write(frame)
                        print(f"Alert: Detected {detected_action}")

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = f'photo/event_{timestamp}.jpg'
                        cv2.imwrite(image_path, frame)

                        with open(csv_file_path, mode='a', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow([timestamp, detected_action])

                       
                        send_email(image_path, detected_action)

                        if alert_counter >= alert_duration:  
                            print("Alert condition met for 5 seconds.")
                    else:
                        alert_counter = 0
                        alert_active = False
                else:
                    color = (0, 255, 0) 
        else:
            color = (0, 255, 0)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            if alert_active:
                color = (0, 0, 255)  
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                cv2.putText(frame, detected_action, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                color = (0, 255, 0) 
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow('Frame', frame)

        prev_frame = gray_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()