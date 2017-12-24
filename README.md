**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Raw_image.png "Visualization"
[image2]: ./images/Image_of_labels.png "Visualization"
[image3]: ./images/Count_in_test.png "Visualization"
[image4]: ./images/Count_in_valid.png "Visualization"
[image5]: ./images/Count_of_train.png "Visualization"
[image6]: ./images/Gray_image_augment.png "Visualization"
[image7]: ./images/New_images.png "Visualization"
[image8]: ./images/True_label_of_new_images.png "Visualization"
[image9]: ./images/Top_5_prob.png "Visualization"
[image10]: ./images/Feature_CNN_1.png "Visualization"
[image11]: ./images/Feature_CNN_2.png "Visualization"

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 43799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32 x 32 x 3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The first 49 images in
the training dataset are plotted in the following figure.


![alt text][image1]

Here we plot one image form label number 0 to 42 as an example.
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how images are pre-processed.

As a first step, all the images are coverted to gray scale becuase the
classifier should be invariant to colors. For example, each label has unique
pattern that does not matter whick color it is.  In addition, converting
colored images to gray cale images also reduces the features size by 66% as the
channel number reduces from 3 (RGB) to 1.

After converting to gray scale images, the local contrast of the images are
also adjusted using the CLAHE function in OpenCV package.  After this step, the
pattern is bettered distinguished expecially when the background or the
surrongding enviroment is too dark.

Then the images are zoomed in to reduce the impact from the surronding environemnt by using the central 28 x 28 pixels, instead of
32 x 32 pixels.

The final step is normalization, which converts all the values to the range of
-1 to 1.
Here is an example of the first 49 traffic sign after pre-processing.

![alt text][image6]
The difference between the original data set and the augmented data set is the following:

* The original dataset are RGB images with size of 32 x 32 x 3. The
  augmented images is 28 x 28 x 1.
* The patterns in the augmented images are more easily distinguished becuase
  the contrast has been adjusted.
* The aumented images also removes some noisy environment by cropping the
  central part of the original images.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)

The final model for the classifier is based on convolution neutral network. The
structure the the model is shown below:

| Layers       | Size         | Activation   |
|:------------:|:------------:|:------------:|
|Input|28x28x1||
|Cov1|28x28x6, same padding|elu|
|Max pooling|14x14x6||
|Cov2|14x14x16, same padding|elu|
|Max pooling|7x7x16||
|L1, fully connected| 784|elu|
|Dropout|||
|L2, fully connected| 400|elu|
|Dropout|||
|L3, fully connected| 43| softmax_cross_entropy|


#### 3. Describe how you trained your model.

During training, we reduce the mini-batch size to 64 so that the whole process can be done on my laptop. The other parameters used in the final version of the model is:

* Conv1: width/height = 5, strike space = 2, zero-padding
* Conv2: width/height = 5, strike space = 1, zero-padding
* Max-pooling: width/height = 2, strike space = 2, zero-padding
* Dropout rate: 60%
* Optimization algorithms: Adam-optimizer
* Learn rate: 0.0005
* Mini-batch size: 64
* Epochs: 100


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.


The training and validation accuracy during training in the first 100 epoches
can be seen in the [notebook html file](./Traffic_Sign_Classifier.html)

The final model results were:
* training set accuracy of 100%.
* validation set accuracy of 94.9%.
* test set accuracy of 95.3%.

The architecture was modified from the well-known LeNet-5 CNN. Compared to
conventional neural network, convolution neural netwok has show success
detection of pattern recognization as it applies share weights algorithm. By
scanning the image repeatly with the same small neuron, the location of
patterns in the raw image does not affact the detection and reconization
process.

In addition, the whole training and testing process is done on a person laptop
(2012 Macbook Pro with 2.9 GHz Intel Core i7, 8 GB 1600 MHz DDR3), this
architecture is simply but powerful enough to acheive 95% accuracy in less than
1 hour of training process.

### Test a Model on New Images

#### 1. Choose six German traffic signs found on the web and provide them in the report.
Six German traffic signs are downloaded from the web. First we use powerpoint
and adobe software to crop the traffic signs out of the raw images. Then these
five images are resized to the shpe of 28 x 28 x 3, as shown in the figure
below:

![alt text][image7]

These new images are further processed using the same method as in the
pre-processing stage. Meanwhile, the true label of these new images are
created using the csv file of German Traffic signs.


![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Next the pre-trained model was used to predict the labels for these six new
images.  The accuracy is 100%.
The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The top 5 probability for each new images are plotted in the figure below. The
model accuratly predicted all of the six images with almost 100% for the
corrected label.


![alt text][image9]


### Visualizing the Neural Network
Here the feature maps in the two convolution layers for the six images are
examined.
Below is all the six filter maps in the first convolution layer. The six
filter maps clearly displace different features of the maps.

![alt text][image10]


Below is all the 16 filter maps in the second convolution layer. The dimention
of the second filter map is 7 x 7.  It is not very clear which filter maps
represent.  But each of the filter maps seems very distinct features.

![alt text][image11]


### Limitation and future improvement

Although the model shows very good accuracy in the study, the testing accuracy
drops down when the traffic signs are not centered cropped from the initial images.

The above model can be further improved by appling an advanced pattern recognization architecture such as ResNet
with a bounding box detection method.
