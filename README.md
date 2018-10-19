# Neural Network car-following model

In this code a feed-forward neural network is trained data 
from a traffic jam experiment conducted by Sugiyama et al. (2008) (http://iopscience.iop.org/article/10.1088/1367-2630/10/3/033001/meta). In this experiment cars drove in a circle of 230m. Two experiments were done (case 1 and 2) with 22 or 23 cars, respectively. With the script "preprocessing.py" the headway, velocity and acceleration is calculated for each car. The neural network is given the task to predict the acceleration of the subsequent data point based on the headway and the velocity of itself (and if wanted from cars ahead).

## Requirements
Keras
Numpy
matplotlib
pandas

## Data
The position data of the cars in the experiment  can be downloaded here:
http://iopscience.iop.org/article/10.1088/1367-2630/10/3/033001/meta
