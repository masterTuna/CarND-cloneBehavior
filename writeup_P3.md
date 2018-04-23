# **Project Behavioral Cloning** 

In this project, I am using NN to clone the driving behavior. For different conditions(car position and road), the regression will predict the suitable angles.

[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./image_cut.png "Image after cut"
[image3]: ./train_history.png "Training/Validating History"
[image4]: ./original.png "Normal Image"
[image5]: ./flipped.png "Flipped Image"
[image6]: ./dist.png "Image angle distribution"

### Files
My project includes the following files:
* model.py Script to create and train the model
* drive.py Diving the car in autonomous mode
* model.h5 Trained convolutional neural network 
* writeup_P3.md Wrap-up of all the implementation
To start the autonomous simulation, use:
```sh
python drive.py model.h5
```

### Image examples
#### 1. Driving in the center of the road
#### 2. Left turn
#### 3. Right turn

### Model Architecture and Training Strategy

#### 1. Architectures
I am using the architecture proposed by NVIDIA. The captured image is 160 x 320 x 3. So in Keras, the layers are arranged like this:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Color image                         | 
| Lambda         		| normalize and centralize the mean             | 
| Cropping				| Crop image: 66x320x3 |
| Convolution2D      	| 5x5, 24 filters, valid padding, 2x2 stride, relu activation|
| Dropout               | keep_prob=0.5|
| Convolution2D			| 5x5, 36 filters, valid padding, 2x2 stride, relu activation|
| Dropout               | keep_prob=0.5|
| Convolution2D			| 5x5, 48 filters, valid padding, 2x2 stride, relu activation|
| Dropout               | keep_prob=0.5|
| Convolution2D			| 3x3, 64 filters, valid padding, 1x1 stride, relu activation|
| Convolution2D			| 3x3, 36 filters, valid padding, 1x1 stride, relu activation|
| Flatten				| flatten to 2112|
| Dense					| from 2112 to 100 |
| Dropout               | keep_prob=0.5|
| Dense					| from 100 to 50|
| Dropout               | keep_prob=0.5|
| Dense					| from 50 to 10|
| Dense					| from 10 to 1|
| Loss					| mse|
 
Visualization of the architecture is:
![alt_text][image1]

* I am using Keras Lambda layer to normalize the data and central the mean to 0, followed by Cropping2D layer, which cuts off the image such that only the zones of interest left. Here from top, 70px was cut and from bottom 24px was cut. An example after cut is as follows:
![alt_text][image2]

* Convolution2D layers were adopted in this model with relu activation. The same kernel size are used as the model in End to end learning for self-driving cars from NVIDIA. 


#### 2. Model optimizer and epochs

My model used an adam optimizer and mean square error as loss function. The epochs was set to 6 which seemed to work well. From the history of optimization, the error kept decreasing over the epochs
![alt_text][image3]

#### 3. Training and validating data

* For the first trace, the training data is from the example data of the course. I had some concerns about how effective it would be. But after the training, it turns out working well for the first track. 

* After reviewing the center image data, I found that a lot of data actually has 0 steering, compared to less data that has meaningful "turning" information. So to augment data, I chose to take advantage of images from all three cameras.

* Data augmentation. As there are three different camera views of data, I am using the following ways to augment data.
	- To form a batch of data, each time I will toss a coin to decide which view to use(Center/Left/Right).
	- When the image view is read in, apply the adjust angle to current steering. The adjust angle is .25, so for image from left view, the new steering would be orignal steering plus .25 while for image from right view, the new steering would be orignal minus .25.
	- And I will toss the coin again to decide if operation of flipping the image should be applied. 
	- For each epoch, I am using twice as much data as I have. Since for each data, it could be either selected of different view or be flipped, the chance to have duplicate data is low.
So an example of orignal image and flipped image is:
![alt_text][image4]
![alt_text][image5]
So finally the distribution of the data looks like:
![alt_text][image6]

* All the data was shuffled and splitted. 20% of the data was used for validation while all the others were used for training.

* Overfitting issue. I am using NVIDIA GTX1060 to train the model. Within a few epochs, the training set loss turns to be super good, while the validation set error starts increasing. To address this issue, I added dropout layer and set the keep_prob to 0.5. I tweaked the dropout layers and the number of epochs to avoid the overfitting. Finally it works well on validation set.

#### 4 Test the model
During the training, the validation error was best when 4 is chosen for epochs. Apply the model to autonomous driving by:
**python drive.py model.h5**
the car ran along the road without drifting to the side or off the road. 
The video is named: video.mp4


