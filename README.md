# Human-Pose-Estimation-and-Action-Recognition

## Introduction
Human pose estimation the predicting poses of human body parts and action recognition is recognizing the human's actions. In this project, I used Mediapipe for human pose tracking and it predict four types of move actions, there are right, left, up, and down.

## How to use
This project includes two files,

<ul>
<li><a href='https://github.com/JafirDon/Human-Pose-Estimation-and-Action-Recognition/blob/main/Data%20Collection.ipynb'>Data Collection</a></li>
<li><a href='https://github.com/JafirDon/Human-Pose-Estimation-and-Action-Recognition/blob/main/Human%20Pose%20Action%20Recognition.ipynb'>Human Pose Action Recognition</a></li>
</ul>

## Data Collection

In this notebook, code is used for labeling our actions and stored in a CSV file. It also provides some EDA analysis for pose key points data.


That code helps to estimate the human pose landmarks
```
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    results = holistic.process(image)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 
                                 
```           


## Human Pose Action Recognition

In this notebook, code is used for predicting human poses and actions. First, we train to pose key points data using RandomForestClassifier. After training is completed, then save the model in a pickle file for an easy-to-load model. Finally, we predict the human poses and actions using the RandomForestClassifier model. It will be live on your webcam using OpenCV.

```
cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        

        results = holistic.process(image)

        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

        try:
            
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            row = pose_row
            
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            
            coords = tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), 
                                                 [640,480]).astype(int))
            
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, 'CLASS', (95,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'PROB', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

```


## License

MIT License





















