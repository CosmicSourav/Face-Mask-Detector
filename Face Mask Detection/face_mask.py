from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plot
import numpy as np
import os

# where the dataset is actually present
dir = r"E:\python project\face mask detection\dataset"
# the data in the category is actually the folder names that are present in the directory path
imageCategory = ["with_mask", "without_mask"]

# so that we have context on what is happening
print(" Stay Clam. The images are being loaded.....")

# append all the image array inside the data list
data = []
# append the labels with or without mask corresponding to the data list
label = []

# looping through the categories
for imageCategory in imageCategory:
    # joining the directory and category
    # first looping through with mask then without mask
    path = os.path.join(dir, imageCategory)
    # list all the images inside the directory
    for images in os.listdir(path):
        # join the path of particular with mask to the corresponding image
        image_path = os.path.join(path, images)
        # loading the image and setting the height and width of all the images as 224, 224 as in the target size
        # coming from keras.preprocessing.image
        image = load_img(image_path, target_size=(224, 224))
        # convert the image to array's
        # coming from keras.preprocessing.image
        image = img_to_array(image)

        image = preprocess_input(image)

        # appending the array to the data list
        data.append(image)
        # appending label array to the label list
        label.append(imageCategory)

# we have all the data as numerical values but the labels are not
# that's why converting the alphabetical labels to array using one hot encoding
# using the label binarizer method which is coming from sklearn.preprocessing
LB = LabelBinarizer()
label = LB.fit_transform(label)
# converting the label values to categorical values
label = to_categorical(label)

# ones the label values are converted to numerical values
# now we need to convert those values into numpy arrays
# also the data values to numpy arrays
# because only with arrays our deep learning model will work
data = np.array(data, dtype="float32")
label = np.array(label)

# splitting the training and testing data

trainX, testX, trainY, testY = train_test_split(data, label,
                                                test_size=0.20, stratify=label, random_state=42)

# specifying the learning rate, epochs and the batch size
# When the learning rate is less, our loss gets calculated properly
# here the learning rate is 0.0001
learningRate = 1e-4
epochs = 20
BS = 32

# image data generator is basically data augmentation
# it creates many images from a single image by adding various properties like rotating the image,
# shifting the image, changing height and width, etc. so that we can create more data with this

aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

# using the mobilenetv2 we care basically creating a base model here here imagenet is there are some pretrained
# models only for images, so that when we use imagenet those weights will be initialized for us and it will give us
# better results include top is whether include fully connected layer at the top of our network input tensor is the
# shape of the image that is going through and 3 is the three channels f the image that is the RGB becuase we are
# inputting colored images
PreModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

Finalmodel = PreModel.output
# average 2d pooling reduce the size of data, number of parameters, amount of computation needed and also controls
# overfitting
Finalmodel = AveragePooling2D(pool_size=(7, 7))(Finalmodel)
# flatten will flatten the multi dimensional input tensor into a single dimension
Finalmodel = Flatten(name="flatten")(Finalmodel)
# dense is used to create fully connected layers in which every output depends on every input
# relu is a goto activation function for non linear use cases
# dense layer is added using 128 neurons
Finalmodel = Dense(128, activation="relu")(Finalmodel)
# drop out is used to avoid overfitting of our model
Finalmodel = Dropout(0.5)(Finalmodel)
# this is the final output layer so 2 is used one is for with mask and one is for without mask
# here softmax activation function is used
Finalmodel = Dense(2, activation="softmax")(Finalmodel)
# place the head FC model on top of the base model (this will become the actual model we will train)
# it accepts two parameters one is the input which will be the base model input
# and one will be the output which will be the head model
model = Model(inputs=PreModel.input, outputs=Finalmodel)
# initially we need to freeze the layers in the base model so that they will not be updated on the first training
# process. because they are just a replacement for our convolutional neural network
for layer in PreModel.layers:
    layer.trainable = False

# optimizing the model
# adam is a goto optimizer just like relu
optimizerUsed = Adam(lr=learningRate, decay=learningRate / epochs)
# giving the parameters for compiling the model using the binary crossentropy as the loss function
# and here we are going to track only the accuracy metrics
model.compile(loss="binary_crossentropy", optimizer=optimizerUsed, metrics=["accuracy"])

# training the head of the network
print("training head....")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=epochs
)

preIndexes = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the label with corresponding largest
# predicted probability
preIndexes = np.argmax(preIndexes, axis=1)

# printing the classification report
print(classification_report(testY.argmax(axis=1), preIndexes, target_names=LB.classes_))

# H5 is a file format to store structured data, it's not a model by itself. Keras saves models in this format as it
# can easily store the weights and model configuration in a single file.
print("saving mask detector model....")
model.save("mask_detector.model", save_format="h5")

# now at last plotting the training loss and accuracy
N = epochs
plot.style.use("ggplot")
plot.figure()
plot.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plot.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plot.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plot.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plot.title("Training loss and Accuracy")
plot.xlabel("Epoch #")
plot.ylabel("Loss/Accuracy")
plot.legend(loc="lower left")
plot.savefig("plot.png")






