# Face-Mask-Detector
The purpose of the project “Face Mask Detection Using Machine Learning” is to  create a tool that identifies a person, or a group of people is wearing mask or not in the real-time. Due to COVID, wearing a face mask is must in  order to stay safe. As the country starts going through various stages of  reopening, face masks have become an important element of our daily lives to  stay safe. Wearing face masks will be required in order to socialize or conduct  business. So, this application utilizes a camera to stream live video and then pass it though the model to detect if a person is wearing a mask or not.

Tensorflow, Keras, and OpenCV are three Python libraries used to develop and build the model presented here. The convolutional neural network model we utilised was MobileNetV2. Transfer Learning is the way of employing MobileNetV2. Transfer learning is the process of utilising a previously trained model to train your current model and obtain a prediction, which saves time and simplifies the process of training various models. The hyperparameters: number of epochs, learning rate, and batch size are used to fine-tune the model. The model is trained on a set of photos divided into two categories: with and without a mask. There are 1915 photos with masks and 1918 images without masks in the collection.

![Final Year project new](https://user-images.githubusercontent.com/25551233/224434814-0bbfb882-cedd-4266-b304-0854cb6488b0.jpg)

# Result
After 20 epochs of training and a batch size of 32, the validation accuracy of this model is 99.8 percent for MobileNetV2.
![Final Year project](https://user-images.githubusercontent.com/25551233/224435076-de37822d-139a-4c97-9065-cecc94350a59.jpg)


# Demo
![Screenshot 2022-03-14 010437](https://user-images.githubusercontent.com/25551233/224432493-b14072c6-c3f8-4522-9e46-b8cc411ab2a0.jpg)
![Screenshot 2022-03-14 010543](https://user-images.githubusercontent.com/25551233/224432544-41ac206f-8506-4a60-9966-43e6155bc7e5.jpg)

# Research Paper Link
https://scholar.google.com/citations?view_op=view_citation&hl=en&user=n4kvsz0AAAAJ&citation_for_view=n4kvsz0AAAAJ:u5HHmVD_uO8C
