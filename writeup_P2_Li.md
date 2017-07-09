#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image.png "Visualization1"
[image2]: ./visualization.png "Visualization2"
[image3]: ./image_cl.png "CLAHE"
[image4]: ./image_cl_zi.png "Zooming in"
[image5]: ./image_cl_cp.png "Cropping"
[image6]: ./image_cl_rot.png "Rotation"
[image7]: ./image_cl_af.png "Affine Transformation"
[image8]: ./visualization_aug.png "Visualization Aug"
[image9]: ./new_images.png "New Images"
[image10]: ./gv_softmax.png "Softmax"
[image11]: ./fm_conv1.png "Feature Maps of Conv1"
[image12]: ./fm_conv2.png "Feature Maps of Conv2"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python with the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. One image shows a random sample in the training set and a histogram shows how the data is distributed in each class. From the histogram we see that the dataset is significantly unbalanced.

![alt text][image1]![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I first did CLAHE (Contrast-Limited Adaptive Histogram Equlization) on all image data. I converted the images to HSV space then perform CLAHE on V layer. Only equalizing the V layer can enhance the contrast without affecting much the image's original color property.

![alt text][image3]

As the dataset is significantly unbalanced, I decided to apply data augmentation to make the dataset almost evenly distributed without losing any original data. I used four approaches to augment the dataset: Zooming in, Cropping, Rotation and Affine Transformation.

Zooming in is simply cropping a patch that is slightly smaller (30 x 30) than the original image. Then resize the image patch back to 32 x 32.

![alt text][image4]

Cropping the original image is to use four 28 x 28 windows along the diagonal then resized the image patch to 32 x 32. That way the number of training examples got increased by a factor of 5. I did not do horizontal or vertical flipping as it would simply introduce unrealistic data.

![alt text][image5]

Slightly rotate the image by 10 degree clockwise and anti-clockwise, respectively. I created three times more of certain training data.

![alt text][image6]

Affine transformation keeps the parallel lines in the original image still be parallel in the output image.

![alt text][image7]

The above data augmentation methods were not applied over all classes. The fewer images one class has, the more augmentation methods were excuted in this class, while making sure that the maximum number of images for a class is about 2500. The data became much more balanced after augmentation.

![alt text][image8]

In addition, I normalized the image data because this is a standard procedure in training neural networks. Since we want to make the weight updates in different sign so that they won't increase/decrease together, we make the average over training set close to zero. We also want to make the learning ratesimilar among weights by scale the inputs so that the variance of each dimension is about the same. In general, data normalization makes learning more efficient.

I decided to generate additional data also because the initial validation accuracy is much lower than the training accuracy. This is a sign of overfitting. Although I used regularizition and dropout, the problem still exists. Since the color does provide cue to classify certain signs, I did not convert the images to grayscale.

To pre-process and add more data to the the data set, I used the Image module in the Pillow library and OpenCV library because they provide the useful functions enabling me to effortlessly create new images for enlarging the dataset. 

The difference between the original data set and the augmented data set is that some signs get "zoomed in", rotated or translated related to the image. 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is still the LeNet used in class. The goal is to maximize its performance. It is consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Fully connected		| outputs 10        									|
| Softmax				|        									|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer(). The batch size is 128 with 100 epochs. I used 0.8 dropout for the last but one layer, L2 regularization for all the FC parameters, and learning rate decay of 0.95 per epoch from the base rate of 0.001 to achieve the best performance.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.823
* validation set accuracy of 0.974 
* test set accuracy of 0.947

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?

I chose the LeNet achitecture.

* Why did you believe it would be relevant to the traffic sign application?

The LeNet achitecutre was used for classifying hand-written digits. The image dimensions and the size of the dataset are very similar to the traffic sign dataset. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
After data augmentation the validation and the test accuracy got improved, but the training accuracy decreased a lot. So, the model appears underfitting the data. But not using dropout and regularization would cause the validation and test accuracy decrease in the experiments. So, I am guessing this is because the data augmentation makes the training set harder while the test set doesn't change. I'd appreciate any thoughts or comments you have on what could the final results look like this. I am using a 4GB GTX 960 GPU, which is not powerful enough for more trials. However, investigating deeper models such as Cifar10 would be a sensible direction for further optimization.

###Test a Model on New Images

####1. Choose at least five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image9]

The first image might be difficult to classify for its arrow direction. It might be hard to be distinguished from the Turn Right sign or Go Straight or Right sign.
The second image might be difficult to classify for the fuzzy number inside the red circle due to the 32 x 32 image's low resolution. It might be hard to be distinguished from the Speed Limit (100km/h) sign.
The third image might be difficult to classify also because of the fuzzy number inside the red circle due to the 32 x 32 image's low resolution. It might be hard to be distinguished from the Speed Limit (20km/h or 50km/h or 60km/h or 80km/h) sign.
The fourth image might be difficult to classify because it is hard to see the sign inside the red triangle clearly due to the 32 x 32 image's low resolution.
The fifth image might be difficult to classify also because it is hard to see the "wild animal" inside the red triangle clearly due to the 32 x 32 image's low resolution.
The sixth image is a stop sign. Stop signs have fairly unique color, shape and pattern cues and therefore might be hard to be misclassified.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead      		| Turn right ahead   									| 
| 120km/h     			| 120km/h 										|
| 30km/h					| 30km/h											|
| Slippery road		      		| Wild animals crossing					 				|
| Wild animals crossing			| Wild animals crossing      							|
| Stop			| Stop      							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3% with only the Slippery Road sign misclassified. This compares favorably to the accuracy on the test set of 94.7%. Most of the signs were classified correctly. However, the model is has trouble predicting Slippery Road sign. The Slippery Road sign is indeed similar to the Wild Animal Crossing sign under poor resolution. It might need deeper model to extract fine features.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is fairly sure that this is a turn right ahead sign (probability of 0.999), and the image does contain a turn right ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99398470e-01         			| Turn right ahead   									| 
| 3.74062569e-04     				| Ahead only 										|
| 9.08179209e-05					| General caution											|
| 4.93655498e-05	      			| Stop					 				|
| 2.33470273e-05				    | Go straight or right      							|

For the second image, the model is fairly sure that this is a speed limit sign (120km/h) (probability of 0.999), and the image does contain a speed limit sign (120km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99999762e-01         			| 120km/h   									| 
| 1.03860458e-07     				| 100km/h 										|
| 5.99910450e-08					| 20km/h											|
| 4.62728345e-09	      			| Beware of ice/snow					 				|
| 3.15736948e-09				    | Slippery road      							|
 
For the third image, the model is fairly sure that this is a speed limit sign (30km/h) (probability of 1.000), and the image does contain a speed limit sign (120km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00         			| 30km/h   									| 
| 4.68362700e-08     				| 70km/h 										|
| 8.39867009e-10					| 80km/h											|
| 5.83129767e-10	      			| 20km/h					 				|
| 1.55509407e-11				    | 120km/h      							|

For the fourth image, the model is relatively sure that this is a Wild animals crossing (probability of 0.839), but the image acturally contains a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.39244068e-01         			| Wild animals crossing   									| 
| 1.60755888e-01     				| Slippery road 										|
| 3.13734386e-11					| Road work											|
| 2.49899472e-12	      			| Double curve					 				|
| 1.92379077e-12				    | Beware of ice/snow      							|

We can see that the model believes the image contains a slippery road with probability of 0.161, which is the correct one the image contains. Compared to the rest, it is pretty close to the largest softmax probability. However, it still fooled the model by making it believe the sign is more like a Wild animals crossing. As mentioned above, this might due to the low resolution of the traffic sign images.

For the fifth image, the model is relatively sure that this is a wild animals crossing sign (probability of 0.999), and the image does contain a wild animals crossing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99999404e-01         			| Wild animals crossing   									| 
| 5.81952520e-07    				| Bicycles crossing 										|
| 8.61149285e-09					| Slippery road											|
| 9.36899031e-11	      			| Beware of ice/snow					 				|
| 2.64952795e-12				    | Double curve      							|



For the last image, the model is relatively sure that this is a stop sign (probability of 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99996185e-01         			| Stop   									| 
| 2.07996891e-06     				| Priority road 										|
| 8.53789402e-07					| Turn left ahead											|
| 5.58018144e-07	      			| No entry					 				|
| 1.54112485e-07				    | No vehicles      							|

Below is the bar charts as the visualizations for softmax probabilities of the 6 new images.

![alt text][image10]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below are the visualization of the feature maps from the first and the second convolutional layer.

![alt text][image11]
![alt text][image12]

For the feature maps of Conv1, we see that they did capture the useful characteristics of the sign such as edges, color, shape and the boundary outline of the sign.

From the second conv layer, the model obtained higher abstraction (generalizatoin) of the features, which also makes sense.

