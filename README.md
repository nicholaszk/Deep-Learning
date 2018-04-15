# Deep-Learning
Using a Fully Convolutional Network (FCN) to train an image data set for semantic segmentation of 3 labels: primary person to follow, other persons, and background.  
  
  
## Project Steps:
### 1. Collecting data 
While the data provided by Udacity was good enough to train a model to a point of finding true positives, I collected my own data for the sake of learning how to collect data and how to navigate the simulator. I collected data as described by the following three scenarios:  
* target in close proximity in a dense crowd (collected while following the target).
* dense crowd with no target (collected while patrolling).
* target in a dense crowd at a distance (collected while patrolling).
  
![Alt text](/images/pathing1.png)
![Alt text](/images/pathing2.png)

### 2. Implementing FCN for Semantic Segmentation
![Alt text](/images/network.png)
The FCN for this project consists of the following three parts:

#### 1. Encoder
The point of the encoder is to extract features. The function uses batch normalization for regularization as well as an activation function known as a ReLU (rectified linear unit) that is being applied to each encoding layer.  
  
#### 2. 1x1 convolutional layer was used instead of fully-connected layers
Typically one might see a fully-connected layer after encoding, but this model uses a 1x1 convolutional layer instead for the following reasons:  
* Flexibility, input images can be any size
* Dimensionality reduction for the layer while preserving the image's spatial information
* More depth with not much more compute cost.

#### 3. Decoder***
This one up-scales (upsample into) the output of the encoder for instance that this will be the same size as the original image. Check 
[Bilinear Upsampling and Decoder Block](/code/model_training.ipynb)  
The convolution layer  added is in order to extract more spatial information from prior layers. In the concatenation step, it is similar to skip connections. Using a skip connection is implemented in order to improve segmentation accuracy [original paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

At the end the activation function Softmax has been applied in order to generate the probability predictions for each of the pixels.

#### Bilinear Upsampling
Bilinear upsampling Bilinear interpolation (an extension of linear interpolation) The key idea is to perform linear interpolation first in one direction, and then again in the other direction, this in order to estimate a new pixel value. In this case, it uses the four nearest pixels.

#### Defined Model
This is the model generated, using the encoder and decoder, both united by a 1x1 Convolution layer:
```
plot_model (model, to_file='model.png', show_shapes=True,show_layer_names=True)
```
![Alt text](/code/model.png)

### 3.Train the model 
for the training it is using the following method:
```
model.fit_generator(train_iter,
                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                    epochs = num_epochs, # the number of epochs to train for,
                    validation_data = val_iter, # validation iterator
                    validation_steps = validation_steps, # the number of batches to validate on
                    callbacks=callbacks,
                    workers = workers)
```
#### chose hyper parameters
Those are the hyper parameters used:
```
learning_rate = 0.001 # value to be multiply with the derivative of the loss function
batch_size = 64 # the batch size is the number of training examples to include in a single iteration
num_epochs = 10# the number of epochs to train for 
steps_per_epoch = 400 # the number of batches per epoch
validation_steps = 100 # the number of batches to validate on 
workers = 2 #maximum number of processes.
```
I trained the FCN with 10 epoch in order to get the expected accurany. I ran the model in my using tensorflow and a GPU. The  batch size and learning rate are linked. If the batch size is too small then the gradients will become more unstable and would need to reduce the learning rate, in this case the batch size is 64. The hyper tuning performed is based on empirical validation. (I have tested with several combination of epochs, batch sizes, learning rates among others.)
This is the final result of epoch 10:

![Alt text](/images/epoch10.png)
```
400/400 [==============================] - 495s - loss: 0.0151 - val_loss: 0.0204
```
### 4.Check the score accuracy
* images while following the target
![Alt text](/images/following_target.png)
![Alt text](/images/following_target1.png)
* images while at patrol without target
![Alt text](/images/patrol_with_targer.png)
* images while at patrol with target
![Alt text](/images/patrol_without_target.png)

**The final grade score is  0.433361966648**

## Future Enhancements
* Adding more training data would be significant, this in order to avoid overfit and get more cases to learn.
* Test with a more architectures, generated by myself or using another proved network like [VGG, ResNet, GoogLeNet and so on](https://medium.com/@siddharthdas_32104/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5) and then update them for Semantic Segmentation as it has been explained in [original paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).. 
