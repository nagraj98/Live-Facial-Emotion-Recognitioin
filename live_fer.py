# import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# paths of models to be used
face_model_path = 'models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/fer2013_cnn2.h5'

# loading models
face_detector = cv2.CascadeClassifier(face_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["Angry","Neutral", "Scared", "Happy", "Sad", "Surprised"]


# starting the camera
cv2.namedWindow('Face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame

    # resizing the frame
    # frame = imutils.resize(frame,width=600)

    grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        grayimg,
        scaleFactor=1.1, #  The image size is reduced by 10% at each image scale.
        minNeighbors=4,  #  To eliminate false positives, min_neighbours is set to a value, generally 3-5
        minSize=(30,30), #  minimum size of a window or a bounding box
        flags=cv2.CASCADE_SCALE_IMAGE
        )

    # loop over the bounding boxes
    for i in range(len(faces)):

        (x, y, w, h) = faces[i]
        
        
        if(i==0):
            # red box around the first face, 
            # we will determine the emotion for only this face
            colour = (0, 0, 255) # BGR Red

            # Extracting the region of interest
            roi = grayimg[y:(y + h), x:(x + w)]
            # when represented in a matrix form, the 'y' or height will correspond to the rows
            # and x or width will correspond to columns
            # as we use (rows, columns) in that order, hence we use (y,x)

            roi = cv2.resize(roi, (48, 48))
            roi = np.expand_dims(roi, axis=(0,3))
            # emotion classifier requires input in the form of array of images of (48,48) size with 1 colour channel
            # hence for 1 image, input shape expected is (1, 48, 48, 1)

        else:
            # green box around all other faces
            colour = (0, 255, 0) # BGR Green

        # draw the face bounding box on the image
        cv2.rectangle(
            frame, 
            (x, y),         # starting pixel, which represents the top left corner
            (x + w, y + h), # ending pixel, which represents the bottom right corner
            colour,         # colour in BGR representation 
            2               # thickness of the line
        )

        # predicting the emotion
        preds = emotion_classifier.predict(roi)
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()

        # looping over all the emotions
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds[0])):

            # construct the label text
            # text = "{}: {:.2f}%".format(emotion, prob * 100)
            mytext = f"{emotion}: {prob*100:.2f}"

            w = int(prob * 300)
            cv2.rectangle(
                canvas, 
                (7, (i * 35) + 5),
                (w, (i * 35) + 35), 
                (0, 0, 255), 
                -1
            )
            cv2.putText(
                canvas, 
                mytext, 
                (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45,
                (255, 255, 255), 
                2
            )
            cv2.putText(
                frameClone, 
                label, 
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (0, 0, 255), 
                2
            )
            
            # cv2.rectangle(frameClone, (x, y), (x + w, y + h),
                            # (0, 0, 255), 2)

        cv2.imshow('Face', frame)
        cv2.imshow("Emotion Probabilities", canvas)
        
    # terminate the program on pressing key 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

