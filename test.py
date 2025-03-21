import cv2
import pickle
import mediapipe as mp
import numpy as np

model_dict=pickle.load(open('./model.p','rb'))  #open saved model
model = model_dict['model'] 
capture = cv2.VideoCapture(0) # open camera to capture video

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.2,min_tracking_confidence=0.2)
#labels_dict={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}
labels_dict = {i: chr(65 + i) for i in range(26)} #dictionary to map labels to alphabets


while True:  # starts infinite loop to capture video

    data_aux=[] 
    ret,frame = capture.read() #read frame from camera  , ret - if frame read successfully and frame - captured image
    
    if not ret:  #check if the frame was read correctly
        print("Failed to capture frame.")
        break

    frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert frame from bgr(opencv) to rgb
    
    results=hands.process(frame_rgb)  #process the frame to detect hands

    if results.multi_hand_landmarks:  #check if hand is detected
       for hand_landmarks in results.multi_hand_landmarks: #loop through each hand detected
           mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style()) #draw hand landmarks using predefined style

       for hand_landmarks in results.multi_hand_landmarks: #loop through each landmark(21 per hand)
               for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    data_aux.append(x) #stores them in data_aux
                    data_aux.append(y) 
       
       if len(data_aux) == model.n_features_in_:  # 21 landmarks × (x, y)
            prediction = model.predict(np.array([data_aux]))  # Fix shape
            prediction_character = prediction[0] if isinstance(prediction[0], str) else labels_dict[int(prediction[0])] # If the prediction is already a letter, use it.
            #Otherwise, convert the predicted number to a letter using labels_dict.
            print(prediction_character)


    cv2.imshow('hand tracking',frame)
    if cv2.waitKey(25) & 0xFF == ord('k'):
        break

capture.release()
cv2.destroyAllWindows()