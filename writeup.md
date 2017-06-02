#**Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image2.jpg "Image"
[image2]: ./examples/image1.jpg "Image"
[image3]: ./examples/startfromright.jpg "Recovery Image"
[image4]: ./examples/midfromright.jpg "Recovery Image"
[image5]: ./examples/finishfromright.jpg "Recovery Image"
[image6]: ./examples/image2.jpg "Normal Image"
[image7]: ./examples/image2flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I began with the Nvidia teams pipeline and added Dropout on 2 of the dense layers.

My model consists of a convolution neural network with three 3x3 filter sizes and depths between 16 and 64 (model.py lines 142-146) 

The model includes RELU layers to introduce nonlinearity (code line 142, 144, and 146), and the data is normalized in the model using a Keras lambda layer (code line 140). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 148 & 151). 

The model was trained and validated on different data sets to ensure that the model was not overfitting . The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 159).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and multiple runs on both tracks.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to get a very small sample of driving, and get my convolutional neural network to correctly identify and drive down the first bit of track.

My first step was to use a convolution neural network model similar to the Nvidia teams model. I thought this model might be appropriate because it fit the data we were working with

With the validation data, I could see the model was overfitting, I put in dropout on two of the Dense layers

I added in MaxPooling and removed a couple Convolution2D Layers from the Nvidia model. I also spent a lot of time tweaking the parameters fed into it. 

Once that was completed, I ran a bunch of runs on both tracks to train the data. See section 3 to see details.

####2. Final Model Architecture

The final model architecture (model.py lines 139-157) consisted of a convolution neural network with the following layers and layer sizes:

1. 3x3 sized 16
2. Max Pooling size 2,2
3. 3x3 sized 32
4. Max Pooling size 2,2
5. 3x3 sized 64
6. Max Pooling size 2,2
7. Flatten
8. Dense 100
9. Dropout .4
10. Dense 50 
11. Dropout .2
12. Dense 10
13. Dense 1


![alt text][image1]

####3. Creation of the Training Set & Training Process

I recorded two laps around the track going counterclockwise (the normal for the simulator), then did one lap going the opposite direction. 

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it lost its way. These images show what a recovery looks like starting from the right and ending in the center :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would add additional training data and strengthen the network. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I preprocessed the data by grayscaling it and scaling it to 50% it's original size (mostly to help my poor aging GPU)

After the collection process, I had 35076 data points, before the augmentation of flipping the images and angles.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by more epochs not yielding a decent return on investment. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I did also get it to complete the second track after some extra training. Video is included but I'm not sure if the current model will still pass it.