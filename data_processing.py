import os  # intraction with files
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  #tensorflow warning messages
import mediapipe as mp  #library for hand tracking
import cv2   #image processing
import matplotlib.pyplot as plt  
import pickle  #saving and loading data

mp_hands = mp.solutions.hands  #hand detection  model
mp_drawing = mp.solutions.drawing_utils   #draw hand landmarks
mp_drawing_styles = mp.solutions.drawing_styles  #drawing styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) #static_image_mode=True means each image is processed independently,min_detection_confidence=0.5 means confidence threshold 

data_path = r"C:\Users\asus\Downloads\ISL"
data=[] # list to store hand landmarks
labels=[] # store coreesponding labels(a-z)

if not os.path.exists(data_path):
    print(f"Error: Directory '{data_path}' does not exist.")
    exit(1)
for dir_ in os.listdir(data_path):
    dir_path = os.path.join(data_path, dir_) #path to each folder

    for img_path in os.listdir(os.path.join(data_path, dir_)):
      dir_path = os.path.join(dir_path, img_path) #img_path is name of image file
      data_aux = [] #list to store landmarks of each image
      x_ = []
      y_ = []
      img = cv2.imread(os.path.join(data_path, dir_, img_path))
      if img is None:
            print(f"Error loading image: {data_path}")
            continue 
      img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converts opencv image(bgr) to RGB
    
      results=hands.process(img_rgb)  #detects hand and get landmarks

      if results.multi_hand_landmarks:  #checks if a hand is detected
            
           for hand_landmarks in results.multi_hand_landmarks:  #loop through each hand detected
               for i in range(len(hand_landmarks.landmark)):
                    x=hand_landmarks.landmark[i].x
                    y=hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

               for i in range(len(hand_landmarks.landmark)):  #re-normalize landmarks
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))    
           
      data.append(data_aux) #add data_aux landmarks to data
      labels.append(dir_)  #add dir_ alpabet to labels
print(f"Total samples collected: {len(data)}")
print(f"Total labels collected: {len(labels)}")
f=open('data.pickle','wb')   #save data and labels to a file
pickle.dump({'data':data,'labels':labels},f)
f.close()