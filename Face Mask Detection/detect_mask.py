from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2


def detect_and_predict_mask(frame, faceNet, maskNet):
    # here we are grabbing the dimension of the frame and then constructing a blob from it
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104, 177, 123))

    # passing the blob through the network and obtain the face detections
    faceNet.setInput(imageBlob)
    detections = faceNet.forward()
    print(detections.shape)
    # initializing the list of faces, their locations, and the prediction of our face mask networks
    faces = []
    locations = []
    predictions = []

    # looping over the detections
    for i in range(0, detections.shape[2]):
        # here we are extracting the confidence which is nothing but the
        # probability associated with the detections
        confi = detections[0, 0, i, 2]
        # here we are filtering out the weak detections and ensuring that the confidence is greater than minimum
        if confi > 0.5:
            # here we are computing the x and y coordinates of the bounding Box for the object
            Box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = Box.astype("int")

            # here we are ensuring the bounding Boxes fall within the dimensions of teh frame
            startX, startY = (max(0, startX), max(0, startY))
            endX, endY = (min(w - 1, endX), min(h - 1, endY))

            # extracting the face ROI, converting it from BGR to RGB channel
            # then we are resizing it pre-processing it and ordering it
            faceCheck = frame[startY:endY, startX:endX]
            faceCheck = cv2.cvtColor(faceCheck, cv2.COLOR_BGR2RGB)
            faceCheck = cv2.resize(faceCheck, (224, 224))
            faceCheck = img_to_array(faceCheck)
            faceCheck = preprocess_input(faceCheck)

            # now adding the face and bounding Boxes to their respective lists
            faces.append(faceCheck)
            locations.append((startX, startY, endX, endY))
    # this line is for we will only make a prediction if at least one face is detected
    if len(faces) > 0:
        # for faster influence we will make batch predictions on all
        # faces at the same time rather than one by one predictions in the above for loop
        faces = np.array(faces, dtype="float32")
        predictions = maskNet.predict(faces, batch_size=32)
    # return a 2-tuple of the face locations and their corresponding locations
    return locations, predictions


# storing the path of  the face_detector model to a variable
protoTextPath = r"E:\python project\face mask detection\face_detector\deploy.prototxt"
weightsPath = r"E:\python project\face mask detection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
# here we are using a method call readNet which is under cv2.dnn and dnn stands for deep neural network
# here we are using the faceNet to detect the face
faceNet = cv2.dnn.readNet(protoTextPath, weightsPath)

# loading the mask detector model from the disk that we have made before
maskNet = load_model("mask_detector.model")

print("Starting video stream....")
# here using the video stream we are accessing the camera and using it
# here src which is source is 0 because we are using the integrated camera of our system
# but if we have more than one webcam than we have to play around with this
vs = VideoStream(src=0).start()

# as we all know that every frame is an image
# and that's why we are looping through each and every frame from the video or live recording
while True:
    # now here we are grabbing the frame from the video and resizing it
    # here the max width of a frame will be 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # here we are sending all the values to the function and detecting whether
    # a person is wearing a mask or not
    (locations, predictions) = detect_and_predict_mask(frame, faceNet, maskNet)
    # looping through the detected face locations and their corresponding locations
    for (Box, pred) in zip(locations, predictions):
        # unpacking the bounding Box and the predictions
        startX, startY, endX, endY = Box
        mask, without_mask = pred
        # giving the color of the Box or rectangle that we are going to draw
        # that is red for no mask and green for mask
        label = "Mask" if mask > without_mask else "No Mask"
        # here we have used BGR that is why if mask is present then the green value is max and rest is min
        # and vice versa
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # adding the probability in  the label
        label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        # displaying the text and the rectangle here frame is the frame that we capture, label is the
        # with mask and without_mask, startX and startY are the coordinated where we want to put the text
        # the we gave the font style then the font scale then color that we had defined before and last the
        # thickness
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 2)
        # here 2 is the thickness of the rectangle
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # showing the output
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    # press and hold the esc button to stop the program
    # 27 is the ASCII of esc
    if cv2.waitKey(2)==27:
            break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()