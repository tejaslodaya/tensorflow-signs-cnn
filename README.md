An algorithm that facilitates communication between a speech-impaired person and someone who doesn't understand sign language using convolution neural networks

Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).

Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).

Here are examples for each number, and corresponding labels converted to one-hot. 
![alt signs_dataset](https://raw.githubusercontent.com/tejaslodaya/tensorflow-signs-nn/master/signs_dataset.png)

Architecture:
1. Input is an image of size 64 x 64 x 3 (RGB), which is normalized by dividing 255
2. Model: 
    ![alt architecture](https://raw.githubusercontent.com/tejaslodaya/tensorflow-signs-cnn/master/images/model.png)
3. The output of last hidden layer gives a probability of the image belonging to one of the six classes
4. RELU activation function. Cross entropy cost. Adam optimizer
5. Mini-batch gradient descent with minibatch_size of 64

The model is CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

Outcome:

1.  Training cost graph-

![alt cost](https://raw.githubusercontent.com/tejaslodaya/tensorflow-signs-cnn/master/images/cost.png)

2.  Train Accuracy - 0.92963 <br>
    Test Accuracy - 0.791667
3.  TODO- to overcome overfitting, add L2 or dropout regularization