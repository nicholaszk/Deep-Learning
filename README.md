# Deep-Learning
Using a Fully Convolutional Network (FCN) to train an image data set for semantic segmentation of 3 labels: primary person to follow, other persons, and background.  
  
  
## Project Steps:
### 1) Data Collection 
While the data provided by Udacity was good enough to train a passing model, I collected my own data for the sake of learning how to collect good data and how to navigate the simulator. I collected data as described by the following three scenarios:  
* target in close proximity in a dense crowd (collected while following the target).
* dense crowd with no target (collected while patrolling).
* target in a dense crowd at a distance (collected while patrolling).
  
![Alt text](/images/pathing1.png)
![Alt text](/images/pathing2.png)

### 2) FCN Implementation for Semantic Segmentation
![Alt text](/images/network.png)
The FCN for this project consists of the following three parts:

#### 1. Encoder
The point of the encoder is to extract features. The function uses batch normalization for regularization as well as an activation function known as a ReLU (rectified linear unit) that is being applied to each encoding layer.  
  
#### 2. 1x1 convolutional layer was used instead of fully-connected layers
Typically one might see a fully-connected layer after encoding, but this model uses a 1x1 convolutional layer instead for the following reasons:  
* Flexibility, input images can be any size
* Dimensionality reduction for the layer while preserving the image's spatial information
* More depth with little more compute cost.

#### 3. Decoder
The decoder is used to upscale our encoded layers into larger images that can weigh the significance of smaller image layers. While the convolution layer is added to extract more spatial information from prior layers, a concatenation step is also used to mimic the effect of a skip connection. Therefore, it is used to improve segmentation accuracy.  
  
A Softmax function generates probability predictions for each pixel.

#### Bilinear Upsampling***
Add explanation****

#### Resulting Model
  
This is the model generated:  
```
plot_model (model, to_file='model.png', show_shapes=True,show_layer_names=True)
```  
![Alt text](/placeholder.png)

### 3) Model Training 
  
#### Hyper Parameter Selection
  
As per Udacity's definitons:  
```
learning_rate = 0.0015  # 
batch_size = 64  # number of training samples/images that get propagated through the network in a single pass
num_epochs = 10  # number of times the entire training dataset gets propagated through the network
steps_per_epoch = 400  # number of batches of training images that go through the network in 1 epoch
validation_steps = 100  # number of batches of validation images that go through the network in 1 epoch
workers = 2  # maximum number of processes to spin up
```
  
General thoughts about why each hyper parameter value was chosen****
   
Final results:

![Alt text](/images/finalepochresults.png)
  
Change this*****
```
400/400 [==============================] - 495s - loss: 0.0151 - val_loss: 0.0204
```
  
### 4.Check the score accuracy
* images while following target
![Alt text](/placeholder.png)
![Alt text](/placeholder.png)
* images while patrolling with no target
![Alt text](/placeholder.png)
* images while patrolling with target
![Alt text](/placeholder.png)

**The final grade score is  0.433361966648**

## Future Enhancements

Add more to this****
  
* Adding more training data would be significant to avoid overfitting.
  
