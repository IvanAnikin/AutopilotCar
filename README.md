

# Autonomous car driving using ML 


## Main structure
![](https://supercoolserver.azurewebsites.net/assets/img/arduino_structure.jpg)

## ML

Machine learning logics are written on Python using Tensorflow and Keras libraries for models operation and OpenCv for Computer Vision image preprocessing and objects detection.

### Agents

####DQN
This agent is created of two similar DQN models: target and q model

#####Models structure
######Inputs: 
1. Resized video frame in all RGB colors
2. BlackAndWhite thresholded image
3. Canny edges frame 
4. Distance

######Output:
Q value for each of the actions (Left, Right, Forward)
![](https://supercoolserver.azurewebsites.net/assets/img/DQN_qnetwork.png)


![](https://supercoolserver.azurewebsites.net/assets/img/arduino_car.jpg)
