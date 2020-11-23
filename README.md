# **Behavioral Cloning** 

## Udacity Project 4 Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

In comparison to course material the key code modificatoons / customizations are in:
* model.py containing neural net model is written from scrach but based on code and hints shown in "Project: Behaviour Cloning" section of the course
* modified generator to do augmentations and append side camera images
* tweaking parameters / network architecture

**Summary:** The final network was trained/validated/tested only on the provided dataset I didn't record any new training data. The network architecture is based on nVidia network as suggested in the course material. I did not use any transfer learning. In order to achieve satisfactory performance meeting assignement objectives I mostly focused on manipulating input data: added normalization, appending side camera images with modified steeting angle, flipping images horizontally. The trained model works only for the first simulator route.


[//]: # (Image References)

[image_1]: ./examples/autonomous_mode_inside.jpg "Model Visualization (inside)"
[image_2]: ./examples/autonomous_mode_outside.png "Model Visualization (outside)"

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following **key files**:
* model.py - containing the script to create and train the model
* drive.py - script for driving the car in autonomous mode (umodified)
* model.h5 - containing a trained convolution neural network 
* README.md - writeup on project coding, challenges and possible improvements

Additional files:
* original_README.md - original project readme file with the description of problem to solve
* video.mp4 - one lap around the test track video; recordeed of simulator in autonomous mode using the trained model.h5 (captured with drive.py)
* video_outside.m4v - same as video.mp4 but with view behind the car (screencapture directly from the simulator)


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**Note:** Due to simplicy of Keras by large the code is similar to what was demonstrated in the Project notes. The main additions are in the input data manipulation and network parameters. I also organized the code into functions to improve its readability and allow faster testing of different blocks.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Implemented model uses almost unaltered nVidia network. The only modifications are input pre-processing and additon of dropout layer after the first Dense layer. Therefore the network includes:
* 2 data pre-processing layers (for normalization and cropping)
* 5 convolutional layers (each followed by RELU)
* 4 dense layers (first followed by dropout and proceeded by flatten)

The network architecture is implemented in lines 146- 162 of model.py. Additionally, lines 123-144 contain functions with other networks I tested.

#### 2. Attempts to reduce overfitting in the model

Lookng at the validation loss/accuracy vs. training the overfitting was not particularly obvious (I did stick to few epochs from the very start tho). However, to mitigate potential overfitting I did add a dropout layer.

I did not test the model beyond the first simulator track. However, when running autonomous mode on this first track with the final version of trained model the car always stayed on the road.

#### 3. Model parameter tuning

For model parameters I did not make any major modifications. Adam optimizer, learning rate, batch size and epoch count are similar as in the project preparation guidelines.

#### 4. Appropriate training data

The bulk of modifications pertains to training data feed into the algorithm


Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Documentation

#### 1. Solution Design Approach

My overall approach followed what was advised in the project preparation instructions - that turned out to be sufficient for having the car drive first assignement task (I did not implement a model for second, optional track).

Some of my network iterations included following in order :
* **implemented all steps same as from the class** - result at best car went til first sharp turn (before the bridge) annd at the end of it fell off the road (this was achieved after moving to nVidia model)
* **[Added augmentation - flipping]** (was removed in-class after adding generator) - car went over the bridge and fell into the water at first right turn
* **[added side camera images, with 0.05 angle correction]** - vehicle seems to stay more in the center than sides of the road but still fails at right turn
* **[modified correction angle for side images to 0.2]** - no change up until after the bridge, car drove off lane on the turn left
* **[added flipped images for side cameras]** - mitigated the problem on the bridge + now the car can make it after the first right turn and complete the entire track!

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 147-160) consisted of simiar layers as the nVidia model. Those have been described earlier on and include: 2 data pre-processing layers; 5 convolutional layers and 4 dense layers.

#### 3. Creation of the Training Set & Training Process

I started experimenting with this assignement using the data provided together with the mockup project files as part of assignement source code. First I decided to focus on improving and pushing the nVidia model as far as I could without moving onto more tedious and obvious pick of collecting more training data.

It turns out that it was possible to make the car stick to the road on the first track without recording any additional training data - therefore I enetuallty did not.

My final dataset was exactly the the samples provided by Udacity for the project: 8036 images, 6428 for training and 1608 for validation. Batch size was almost the same as picked in project preparation however due to my augmentation strategy I had to slightly modify it to 36 images per batch. The final amount of epochs was to 5, beyond that number I didn't observe any improvements.

As required by the assignement I crearted the final video with drive.py script showing the road from drivers perspective (see below picture).

![alt text][image_1]

I also recorded the video from the outside of the car which shows the car keeping to the lane in a more clear way (see below picture).

![alt text][image_2]