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

A FCN is basically a normal CNN (convolutional neural network) where the last fully connected layer has been replaced by a 1x1 convolutional layer. An FCN attempts to capture the entire context of a scene, telling us what objects are in the image (classification) as well as their approximate location in the image for the purpose of semantic segmentation of the image. The FCN for this project consists of the following three parts:

#### 1. Encoder
The point of the encoder is to extract features. The function uses batch normalization as well as an activation function known as a ReLU (rectified linear unit) that is being applied to each encoding layer.  
  
##### Batch Normalization
Instead of just normalizing the inputs to the network, we normalize the inputs to layers within the network. Batch normalization gives us faster network training, higher learning rates, simplifies creating deeper networks, and provides regularization.
```
def encoder_block(input_layer, filters, strides):
    
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```
    
#### 2. 1x1 Convolutional Layer
Typically one might see a fully-connected layer after a CNN, but this FCN uses a 1x1 convolutional layer instead for the following reasons:  
* Flexibility, input images can be any size
* Dimensionality reduction for the layer while preserving the image's spatial information
* More depth with little more compute cost.

#### 3. Decoder
The decoder (deconvolution) is used to upscale (upsample) our encoded layers into larger images that match the original size. A concatenation step is used to mimic the effect of a skip connection, improving segmentation accuracy.  
  
##### Bilinear Upsampling
This process is generally used in the decoder blocks to recover resolution. Different upsampling factors can be used, but I stuck with a factor of 2.
  
##### Concatenation
We lose some information every time we do convolution. To help this, FCNs also seek some activation from previous layers (skip connections) to sum or interpolate together with the up-sampled outputs from decoding. Concatenation is an simple way to achieve this.
  
```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    upsampled = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    concatenated = layers.concatenate([upsampled, large_ip_layer])
    
    # Add some number of separable convolution layers
    first_output_layer = SeparableConv2DKeras(filters=filters,kernel_size=1, strides=1,padding='same', activation='relu')(concatenated)
    
    second_output_layer = separable_conv2d_batchnorm(first_output_layer, filters)
    
    return second_output_layer
```
At the end of the model a Softmax function generates probability predictions for each pixel.
  
Here is the code for our model:
  
```
def fcn_model(inputs, num_classes):
    
    # Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoded1 = encoder_block(inputs, 32, 2)
    encoded2 = encoder_block(encoded1, 64, 2)
    encoded3 = encoder_block(encoded2, 128, 2)
    
    # Add 1x1 Convolution layer using conv2d_batchnorm().
    convuluted1by1 = conv2d_batchnorm(encoded3, 256, 1, 1)
    
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoded1 = decoder_block(convuluted1by1, encoded2, 128)
    decoded2 = decoder_block(decoded1, encoded1, 64)
    decoded3 = decoder_block(decoded2, inputs, 32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(decoded3)
```
  
### 3) Model Training 
  
#### Hyper Parameter Selection
  
As per Udacity's definitons:  
```
learning_rate = 0.005  # 
batch_size = 100  # number of training samples/images that get propagated through the network in a single pass
num_epochs = 42  # number of times the entire training dataset gets propagated through the network
steps_per_epoch = 42  # number of batches of training images that go through the network in 1 epoch
validation_steps = 12  # number of batches of validation images that go through the network in 1 epoch
workers = 4  # maximum number of processes to spin up
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
  
