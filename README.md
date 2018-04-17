# Deep-Learning
Using a Fully Convolutional Network (FCN) to train an image data set for semantic segmentation of 3 labels: primary person to follow, other persons, and background.  

[//]: # (Image References)

[image1]: ./deepLearningPics/pathing1.JPG
[image2]: ./deepLearningPics/pathing2.JPG
[image3]: ./deepLearningPics/trainingcurve.JPG
[image4]: ./deepLearningPics/followcompare.JPG
[image5]: ./deepLearningPics/finalscore.JPG
  
  
## Project Steps:
### 1) Data Collection 
While the data provided by Udacity was good enough to train a passing model, I collected my own data for the sake of learning how to collect good data and how to navigate the simulator. I collected data as described by the following three scenarios:  
* target in close proximity in a dense crowd (collected while following the target).
* dense crowd with no target (collected while patrolling).
* target in a dense crowd at a distance (collected while patrolling).
  
Here are a few images of my spawn and pathing plans:
  
![Alt text][image1]
![Alt text][image2]

### 2) FCN Implementation for Semantic Segmentation

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
  
* Learning rate was determined through trial and error as a compromise between the speedy rate of 0.01 and the better eventual accuracy provided by 0.001.
* Batch size was picked arbitrarily, but I wanted to pick something higher than what a regular PC/GPU could run (I was curious about the ultra-large RAM of AWS's EC2 instance).
* It seemed after much trial and error that approximately 40 epochs would get me a final score of greater than 0.40, so I picked 42 to increase my chances a bit without adding too much compute time.
* Steps per epoch was calculated by dividing the total training set (4131) by the batch size. This yields 41.3, so I rounded up to 42.
* Just as above I divided total validation set size by 100 to get in the ballpark of 12.
* I bumped worker number to 4 because the EC2 server uses a quad-core Xeon processor. This decreased epoch compute time from 72 seconds to 66 seconds.
   
Final results:

![Alt text][image3]
![Alt text][image5]
  
This image compares my FCN's predictions to mask truth for images of the drone following the hero:
  
![Alt text][image4]
  
**The final grade score is  0.436**

## Future Enhancements
  
* Adding more training data would be significant. With much more training data, we could decrease learning rate and drastically increase the number of epochs with much less risk of overfitting.
* I could try adding more layers to the network to capture more context
* With more time to perform trial and error, learning rate could become precisely optimized for a better segmentation score
  
### Applying our model to other objects
If we were to try and use our FCN for tracking other objects like dogs or cars or bicycles, our model architecture would be completely usable. The network layer choices for the encoder, the 1x1 convolutional layer, and the decoder would work fine. We wouldn't have to make any changes to our filter and upscaling parameters or where we included concatenation steps. The entire model would be functional.
  
However, our current data and weights would NOT work. We would need to throw out all our images of people and collect new data (images) of whatever object we wanted to track along with whatever background and other objects would need to be segmented. In the example of a car, we would need thousands of pictures of the particular hero car that we want to isolate as well as images of several other cars. Just as in this project we would need images with just the hero car, images with only non-hero cars, and images with both the hero and other cars. With our new data, we would need to retrain our model and produce a new set of weights. The FCN would then work on cars.
  
