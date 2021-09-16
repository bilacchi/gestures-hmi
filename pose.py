#%%
import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

#%%
def mediapipe_detection(image:np.ndarray, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Change colors' channels to RGB based
    image.flags.writeable = False
    results = model.process(image) # Perform prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Change back channels to BGR based
    return image, results

#%%
def draw_landmarks(image:np.ndarray, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

#%%
def draw_styled_landmarks(image:np.ndarray, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color = (80,110,10), thickness = 1, circle_radius = 1), 
                             mp_drawing.DrawingSpec(color = (80,256,121), thickness = 1, circle_radius = 1)) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color = (80,22,10), thickness = 2, circle_radius = 4), 
                             mp_drawing.DrawingSpec(color = (80,44,121), thickness = 2, circle_radius = 2)) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color = (121,22,76), thickness = 2, circle_radius = 4), 
                             mp_drawing.DrawingSpec(color = (121,44,250), thickness = 2, circle_radius = 2)) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color = (245,66,230), thickness = 2, circle_radius=2)) 
    
#%%
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

#%%
DATA_PATH = os.path.join('MP_Data')
no_sequences = 50
sequence_length = 30
start_folder = 50


#%%
def buildModel(nclasses:int=3, weights:str=None):
    model = tf.keras.models.Sequential([tf.keras.layers.LSTM(64, return_sequences = True, activation = 'relu', input_shape=(30,1662)),
                                        tf.keras.layers.LSTM(128, return_sequences = True, activation = 'relu'),
                                        tf.keras.layers.LSTM(64, return_sequences = False, activation = 'relu'),
                                        tf.keras.layers.Dense(64, activation = 'relu'),
                                        tf.keras.layers.Dense(64, activation = 'relu'),
                                        tf.keras.layers.Dense(32, activation = 'relu'),
                                        tf.keras.layers.Dense(nclasses, activation = 'softmax')])
    
    if weights is not None:
        model.load_weights(weights)

    return model

# %%
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA) 
    return output_frame

##
def runSkeleton():
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show to screen
            cv2.imshow('OpenCV Feed', cv2.flip(image, 1))

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
  
#%%
def runClassification():
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    colors = [(245,117,16), (117,245,16), (16,117,245)]
    model = buildModel(weights='bla.h5')
    
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', cv2.flip(image, 1))

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classification', action='store_false', help="To classify")
    args = parser.parse_args()
    if args.classification:
        runSkeleton()
    else:
        runClassification()
