# **Traffic Sign Recognition** 

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image_visualization]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to keep 3 color channels of the images because they may be used as feature maps. Experiments will be done to check their impact on the image recognization accuray. 

As the second step, I normalized the image pixel data from range (0,255) to range(0,1). Black is 0, white is 1. This normalized process helps to increase accuray.  

I decided not to generate additional data because increasing the dimensions of convolutions already generated good enough accuracy. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3x10  | 1x2x2x1 stride, same padding, outputs 28x28x10 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 5x5xx1x30	| 1x2x2x1 stride, output 10x10x30    			|
| RElU                  |                                               |
| Flatten        		| output: 750  									|
| Fully connected 120	| output 750x120   							|
| Fully connected 43	| output 120x43									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I first used epochs as 20, and dropout as 0.75. The accuracy was not satisfactory, which was 88%. Then I used epochs as 30, and dropout 0.85. The accuracy was increased to 92%, still a little below the expectation. Last, I used epochs 50, dropout as 1. The accuracy for validation increased to 96%, and accuracy on the test data turned out to 94.7%, which met the requirement. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 96.7% 
* test set accuracy of 94.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I first used exact LeNet-5 architecture with same dimensions and layers. 
* What were some problems with the initial architecture?
The initial accuracy was only about 88%. I thought that might be the model was oversimplified.  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I increased the first convolution layer to 10 depths, and second to 30 depths, respectively, to match the complexity. In addition, the full connected layers are also increased from 400 to 750. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I tried dropout layers from 0.6 to 1, and found dropout = 1 generated the highest accuracy. 

If a well known architecture was chosen:

My architecture was from LeNet 5 with modification to reflect the uniqueness of the traffic signs. The traffic signs are more complicated than just grayscaled numbers/characters so I increased the first convolution layer to 10 depths, and second to 30 depths, respectively, to match the complexity. In addition, the full connected layers are also increased from 400 to 750. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Please referring to the results above. 
 

### Test a Model on New Images

I randomly selected 30 images from German traffic website, and used these data to test my model. The result was 96.7% accuracy, very promising. Please see my jupyter notebook for details. 