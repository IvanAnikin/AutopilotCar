

# Autonomous car driving using ML 


## Main structure
<img src="https://supercoolserver.azurewebsites.net/assets/img/arduino_structure.jpg" width="300"/>

## Car
### Hardware structure
1. Arduino Mega
2. ESP 32 camera and Wi-Fi module
3. Ultrasonic distance sensor
4. Wheels IR speed sensor - calculating exact distance
5. 2 servos robotic arm - camera rotation
### Software part
You can find the code of the Arduino car [here](https://github.com/IvanAnikin/AutopilotCar/blob/main/Arduino%20Code/sketch_Car-datagatherer.ino)
## ML

Machine learning logics are written on Python using Tensorflow and Keras libraries for models operation and OpenCv for Computer Vision image preprocessing and objects detection.

### Types
Type of the model is defined by rewards the model was given to specify purpose of actions

#### Explorer
This type is focused on exploring as bigger territory as possible. 
Positive reward was given for image frames difference and new objects detected. And negative for getting too close to some obstacle.

#### Detective
The model was trained in a same way as the explorer with an addittion of searching for a specific object in frames. Searching is made by looking for similar hotpoints with ones in a folder with example images of the wanted objects.
 <img src="https://supercoolserver.azurewebsites.net/assets/img/ml_car_detective.jpg" width="500"/>

#### Follower
This type is different in it's base, because the action decision, doesn't come from the neural network, but from an algorithm, that detects the object(face) and moves to the direction of the object. I've made it in different variations:
  - Looking for a face in the frame from car and moving to the direction of the face movement
  - Choosing a specific object and moving to the direction of the object movement
  - The following was made with webcam movement only or with the movement of the whole car, using wheels motors rotation 
  <img src="https://supercoolserver.azurewebsites.net/assets/img/ml_car_face_following.gif?raw=true" width="500"/>

### Agents

#### DQN
This agent is created of two similar DQN models: target and q model

##### First training

It took around 20 hours to train the model on 10 000 steps dataset (10 000 frames, distances, random action and other infromation calculated from these three) locally on my computer
Model weights are saved [here](https://github.com/IvanAnikin/AutopilotCar/tree/main/Model_Weights/%5B%5B0%2C%200%2C%201%5D%2C%20%5B1%2C%201%2C%201%2C%201%2C%201%5D%5D) in .h5 format

##### Models structure
###### Inputs: 
1. Resized video frame in all RGB colors
2. BlackAndWhite thresholded image
3. Canny edges frame 
4. Distance

###### Output:
- Q value for each of the actions (Left, Right, Forward)

 <img src="https://supercoolserver.azurewebsites.net/assets/img/DQN_qnetwork.png" width="500"/>
