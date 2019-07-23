# tf_nextactivity
predicting next activities using lstm neural network

This is a basic character level lstm neural network.
It gets an activity and predicts what is the next activity.

## dependencies
* python 3.6~
* tensorflow
* pyvis

## usage
first of all, you should set up the configuration of the neural network and data. It can be done by modifying train.py's parser or you can directly put argument to it.

* py train.py
run the training phase of the neural network

* py sample.py
there are two functions in the sample.py, first is predict without sequence and second is with sequence.
choose one and call in main
