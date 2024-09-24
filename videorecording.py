import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from emailnotify import *

model = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

actions = ['choking', 'slapping', 'none']

frame_buffer = []
BUFFER_SIZE = 16

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

    print("Probabilities:", probabilities.numpy())

    max_index = tf.argmax(probabilities, axis=-1).numpy()[0]

    if max_index >= len(actions):
        print(f"Warning: max_index {max_index} is out of bounds")
        detected_action = 'none'
    else:
        detected_action = actions[max_index]

    return detected_action

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if ret:
        processed_frame = preprocess_frame(frame)
        frame_buffer.append(processed_frame)

        if len(frame_buffer) == BUFFER_SIZE:
            detected_action = detect_behavior(frame_buffer)
            frame_buffer = []

            if detected_action in ['covering_face', 'slapping', 'punching']:
                out.write(frame)
                print(f"Alert: Detected {detected_action}")

                send_email()

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()