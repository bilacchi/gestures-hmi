#%%
import os
import cv2
import argparse
import numpy as np
import mediapipe as mp
import tensorflow as tf

from sklearn.model_selection import train_test_split
from pose import extract_keypoints, draw_styled_landmarks, mediapipe_detection, buildModel

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

#%% Set Params
camnum = 1
DATA_PATH = './data' # Data directory 

try: os.mkdir(DATA_PATH)
except: pass

actions = np.array(['finger_swipe_left', 'finger_swipe_right', 'finger_swipe_up',
                    'finger_swipe_down', 'finger_point', 'stop', 'arm_angle_down',
                    'arm_swing', 'arm_swipe_up', 'arm_swipe_down', 'arm_still_right', 
                    'arm_still_left', 'none', 'doing_nothing']) # Target actions

nseries = 50 # Number of samples
nbframes = 30 # Number of frames

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
                    ret, frame = cap.read() # Get frame from webcam
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

    x_train, x_test, y_train, y_test = train_test_split(dataSeries,labelSeries, train_size=0.8, random_state=42)
    

    model = buildModel(len(actions)-1)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top@1'),
                                                                            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top@3')])
    callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=.2, verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir='./Logs')
        ]

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2000, callbacks=callbacks)
    model.save('bla.h5')


def run():
    model = tf.keras.models.load_model('bla.h5')
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(1)
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
                #image = prob_viz(res, actions, image, colors)
            
            image = cv2.flip(image, 1)
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
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
    
    print(f'{args.run=}')
    print(f'{args.train=}')
    print(f'{args.sampling=}')
    print(f'{args.complete_sampling=}')
    
    if args.run:
        run()
    elif args.train:
        train()
    elif args.sampling:
        sampleColletor(args.complete_sampling)