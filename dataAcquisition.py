#%%
import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
import tensorflow as tf

from collections import deque
from sklearn.model_selection import train_test_split
from pose import extract_keypoints, draw_styled_landmarks, mediapipe_detection, buildModel, prob_viz

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

#%% Set Params
camnum = 1
DATA_PATH = './data' # Data directory 

try: os.mkdir(DATA_PATH)
except: pass

#actions = np.array(['finger_swipe_left', 'finger_swipe_right', 'finger_swipe_up',
#                     'finger_swipe_down', 'finger_point', 'stop', 'arm_angle_down',
#                     'arm_swing', 'arm_swipe_up', 'arm_swipe_down', 'arm_still_right', 
#                     'arm_still_left', 'none', 'doing_nothing']) # Target actions

actions = ['none', 'arm_swipe_left', 'arm_swipe_up', 'arm_still_left', 'arm_still_right', 'arm_angle_down', 'stop']

nseries = 50 # Number of series per action
nbframes = 30 # Number of frames per series
nbstabil = 10 # Number of frames to stabilize

#%%
for action in actions:
    if action not in os.listdir(DATA_PATH):
        os.mkdir(os.path.join(DATA_PATH, action))
     
#%%
def sampleColletor(complete:bool=False):
    cap = cv2.VideoCapture(camnum)
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:    
        for action in actions:
            actionPath = os.path.join(DATA_PATH, action)
            offsetSerie = len(os.listdir(actionPath))
            print(f'{action=}')
            print(f'{offsetSerie=}')
            print(f'{(offsetSerie % nbframes == 0)=}')
            if complete and (offsetSerie % nbframes == 0): continue
            for serie in range(nseries): 
                window = list()
                for frameNumber in range(nbframes + 1):
                    _, frame = cap.read() # Get frame from webcam
                    image, results = mediapipe_detection(frame, holistic) # Make detections
                    draw_styled_landmarks(image, results) # Draw landmarks on current frame
                    
                    if frameNumber == 0: # Await the first frame
                        image = cv2.flip(image, 1)
                        cv2.putText(image, f'{action} - #{serie}', (len(image)//2, len(image)//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        keypoints = extract_keypoints(results)
                        window.append(keypoints)
                        image = cv2.flip(image, 1)
                        cv2.putText(image, f'{action} - #{serie}', (15,24), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        
                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                # Export keypoints
                np.save(os.path.join(actionPath, str(serie + offsetSerie)), np.array(window))
                        
        cap.release()
        cv2.destroyAllWindows()

#%%
def train():
    labelMap = {label:num for num, label in enumerate(actions)}

    series, labels = [], []
    for action in actions:
        for serie in range(nseries):
            try:
                series.append(np.load(os.path.join(DATA_PATH, action, f'{serie}.npy')))
                labels.append(labelMap[action])
            except: pass
            
    dataSeries = np.array(series)
    labelSeries = tf.keras.utils.to_categorical(labels).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(dataSeries,labelSeries, train_size=0.85, random_state=42)
    

    model = buildModel(len(actions))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top@1'),
                                                                            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top@3')])
    callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=.2, verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir='./Logs')
        ]

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2000, callbacks=callbacks)
    model.save('bla.h5')


def run():
    model = tf.keras.models.load_model('bla.h5')
    queue = deque(maxlen=nbframes)
    queueStabil = deque(maxlen=nbstabil)
    
    threshold = 0.5
    colors = 255*np.random.rand(len(actions), 3)
    
    cap = cv2.VideoCapture(1)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
        while cap.isOpened():

            _, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            
            queue.extend(np.repeat([keypoints], nbframes, axis=0) if len(queue) < nbframes else [keypoints])
            pred = model.predict(np.expand_dims(queue, axis=0))[0]
            
            k = 3 # Get top n predictions
            kth = pred.argpartition(-k)[::-1][:k] # Index of top classes
            val = pred[kth] # Top probabilities

            hist = np.zeros(len(actions)) # Create array of zeros
            hist[kth] = val # Assing probabilities to array
            queueStabil.append(hist) # Append to stabilizatil queue

            avePred = np.array(queueStabil).mean(axis=0) # Calculate mean probability 
            top1 = actions[np.argmax(avePred)] if max(avePred) > threshold else actions[0] # Get top one predicted class
            
            image = cv2.flip(image, 1) # Flip horizontally
            image = prob_viz(avePred, actions, image, colors) # Visualize predictions
            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, top1, (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image) # Show to screen

            if cv2.waitKey(10) & 0xFF == ord('q'): # Break 
                break
            
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('-t', '--train', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-s', '--sampling', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('-c', '--complete_sampling', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    
    if args.run:
        run()
    elif args.train:
        train()
    elif args.sampling:
        sampleColletor(args.complete_sampling)