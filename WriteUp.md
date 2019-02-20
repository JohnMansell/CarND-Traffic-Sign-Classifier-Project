## Data:
> The first step was to explore the data, and become familiar with it. The data was given to us already split into three
groups: training, validation, and testing.
> 
> #### Number of Images
> * Train --  34799  
> * Valid --  4410  
> * Test  --  12630  
> * Dimensions of each Image --  (32, 32, 3)  
> * Number of Uniqe traffic Signs --  43
> 
> So that I could better understand the data, I wrote code to show one sample picture of each type of traffic sign.
![](images_and_label_data/writeup_images/samples.png)

## Distribution of Data:
> The data given to us was not evenly distributed, there were significantly more examples of some classes of traffic 
sign than of others. This can be an issue because it teaches the computer that a certain type of traffic sign is more 
likely to be the correct guess simply because there is an abundance of those examples in the training data. To correct 
this, I created duplicates of the images that were under-represented, and then modified those duplicates sligtly so that
 they would not be exact coppies. This was to help the model generalize.  
> 
> ##### The duplicate copies were put through 3 different transformations:
> * random translate    -- move the image around in its 32x32 container
> * random brightness   -- make the image brighter or darker
> * random scale        -- zoom in or out on the image
> 
> ##### One thing that turned out to be vitally important was to:
> * merge the training and validation data
> * duplicate images with lower representation
> * re-split the data into training and validation sets, 80% and 20% respectively.
> 
> The test set needs to represent the real world as closely as possible. Therefore, I didn't include any of the 
augmented data in the set of test images. 
> 
> Below are histograms showing the number of each class of image before and after balancing the data set.
>
![](images_and_label_data/writeup_images/hist_1.png)
![](images_and_label_data/writeup_images/hist_valid_1.png)
![](images_and_label_data/writeup_images/hist_2.png)
![](images_and_label_data/writeup_images/hist_valid_2.png)

## Preprocess the Data:
>#### Convert to Grayscale  
> Intuitively, I did not think that this would help the model. If anything, I though that a HSV or YUV color space would 
be more effective. However, every model of the NN, and every variation of the hyper parameters worked better in 
grayscale than in any other color space.  
It seemed to me that a color space which keeps the color data would be more effective. More information should (in my 
mind) alway yeild better results. I especially expected the HSV and YUV color spaces to perform well, because they 
isolate brightness from the color components. I expected this to be better for the computer to read, since all stop 
signs will be red, but not all of them will have the same brightness. But this was not the case. In all of my testing, 
the grayscale color space performed the best.
> 
> #### Normalize the Data
> Each image was normalized to be between -1 and 1. This helps the computer learn to look at the actual data, rather than 
creating biases if there happens to be a correlation that could skew the biases. For example, if all the pictures of 
stop signs were taken in bright light, and all the pictures of speed limit signs were taken in poor light, the model 
might learn that bright images are more likely to be a stop sign regardless of the content of the image. Normalizing 
the images helps to mitigate those errors.
> 
>#### Shuffle the Data
> The training and validation data were each shuffled together before they were split. The test data was also shuffled, 
but separately from the training and validation data.
> 
>####Gaussian Blur / Hough Line transform(failed)  
> One attempt that I tried was to follow some of the process used in the Finding Lane Lines project. I first made a copy 
of the image. I then followed the same pipeline to convert the coppied image into one that had all the major lines drawn
 on it. Then I stacked the original image with the copy of the image. I was attempting to create a pseudo inception-model
  of the LeNet architecture. Essentially providing two coppies of the image, one with the original information, and one 
  with the lines information already extracted out. The results of this attempt were abysmal. At the time, the model 
  performed with ~90 % accuracy. With the stacked images, the accuracy dropped to about 1%.

# Model Architecture 
> 
> ### Layers
> First I used the LeNet architecture from class. This worked reasonably well, but I wasn't able to get over 89% test 
accuracy. I then tried using the modified LeNet architecture used by Jeremy Shannon in his project, which he moddled 
after the Sermanet/LeCunn traffic sign classification journal article. 
[Jeremy Shannon](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
> 
>|          Layer          |       Size      |
|:-----------------------:|:---------------:|
|  Input grayscale image  |      32x32      |
|          Conv 1         |   [5, 5, 1, 6]  |
|          Conv 2         |  [5, 5, 6, 16]  |
|          Conv 3         | [5, 5, 16, 400] |
| Inception conv2 + conv3 |       800       |
|    Fully Connected 1    |       800       |
|    Fully Connected 2    |       120       |
|    Fully Connected 3    |        43       |
|     Logits (output)     |        43       |
> 
> ### Training
> To train the model, I used the Adam optimizer. I tried every optimizer available from tensorflow. The Adam optimizer
 performed the best for my network. A few of the optimizers were not compatible with the network structure, and I didn't
  create a new structure to test them, so there may be an optimizer that works better for this project.
> ### Process
> #### Impliment LeNet
> * My first step was to reconstruct the LeNet architecture we had been taught in the lessons, and use it with a 32x32 
image. The difficulty here was in undersanding the relationship between input size, stride, depth, and output size. 
Modifying the architecture to use 32x32 images requires manually telling the network what input / output sizes to expect
 for each layer. I looked up a number of totorials on how to calculate the output size of a layer, but ultimately had to
  derive the relationship myself to fully understand how it was working.
> 
> #### Franken-Net
> * Once I had leNett working and I understood how to determine input and output sizes for a layer, I started trying to 
make changes to the network. I tried using up to 6 convolutional layers. I tried getting rid of all the fully connected
 layers. I tried adding convolutions inbetween the fully connected layers. I tried adding and subtracting depth from the
  convolutional layers. Most of these attempts had little positive, or highly negative impacts.
> * I spent a considerable ammount of time trying to impliment my own version of an inception network. I wanted to include
 a version of the image which would have the lines drawn on it similar to the finding lane lines project. I had hoped 
 that giving the computer more information would result in a higher accuracy. Ultimately, I was unable to get this 
 approach to be productive.
> 
> #### Hyper Parameters
> * Durring all this testing, I also tried varying the hyperparameters. In the Adam optimizer, I tried different values 
for 'learning rate' and epsilon. Ultimately, the default values outperformed any other values. I also tuned the color 
channel, epochs, and batch size. I would try different combinations of all the hyper parameters each time I tried a new
 NN architecture, trying to find the best combination.
> 
> #### Color
> * In each variation of the architecture that I tried, my assumption was that more information in each image would be 
beneficial. I assumed color would out perform grayscale. However, on every iteration, grayscale performed as well or 
better than color images.
> 
> #### Data Distribution
> * Throughout this process, the single change that seemed to have the most effect on the accuracy of the model was the 
quantity and quality of the data. Having poorly distributed or poorly duplicated data killed the accuracy of the model.
 Through all my attempts at changing the structure, or tuning the hyper parameters, the thing that really improved the 
 result was the quality, quantity, and distribution of the data.
> 
> ### Results
> My final model results were:
> * validation set accuracy of 99.6%
> * test set accuracy of 95%  
> ** Test accuracy ranged from 93.89 - 95.63 %**
> 
> 
### Test a Model on New Images

> Below are 5 german traffic signes from the web, as well as the networks top 3 best guesses for each image. Above the 
image is the % confidence the network has in its prediction.
> 
>![softmax](images_and_label_data/writeup_images/softmax.JPG)
> #### Accuracy
> Over various attempts, the network had an accuracy of 60-100%. It never failed to correctly identify signs 2, 3, and 5. 
But it was only accurate about 25% of the time with sign 1 (stop sign) and sign 4 (turn right ahead)
> 
> The stop sign may have been difficult because its not "straight on" in the image. The camera isn't looking directly 
at the stop sign. This may be a feature the network is not well equiped to handle, especially in certain cases. When I 
created duplicate data, I applied a random translation, brightness, and scale to the image but not a random warp to 
simulate perspective shift. This may have resulted in the network not being able to generalize to different viewing 
angles for some of the traffic signs.
> 
> The 'turn right ahead' sign may have posed an issue because it has a small sticker of a girl swinging on a swing 
"suspended" from the arrow. I actually chose this on purpose to see how the network would handle this type of situation.
 Obviously, a car driving on the road needs to be able to make the correct decision even if someone has altered the sign.
  A person looking at this would be able to determine which is part of the sign, and which is a sticker placed on top.
   The network was not.
> 
> *The code for displaying the softmax outputs so elegantly was adapted with minimal changes 
from [Jeremy Shannon's](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) code for this same project*
> 
## Visualizing the Feature Map
> Below I've included images of the feature map from the first 3 layers (all convolutional layers) of the network. This
 shows what pixels, or neurons, were excited by the input image. Essentially this should allow us to follow the network
  to see what it's looking for in an image to classify it.

> #### Layer 1
![layer 1](images_and_label_data/writeup_images/feature_map_1.png)  
> In layer 1, we see that the network primarily highlights areas of high contrast. The word "STOP" and the shape of the
 stop sign is clearly visible in each of the convolutional layers.

>#### Layer 2
> ![layer 1](images_and_label_data/writeup_images/feature_map_2.png)  
In Layer 2, the network's activation is more abstract. This layer is a series of 16 layers of convolution of the first 
5 layers of convolution. It looks almost possible to still make out some of the letters of "STOP". Possibly, the second 
layer is looking for the individual letters and shapes. However, its not entirely clear, and the apparent letters could 
just be an example of finding patterns in noise.

>#### Layer 3
> ![layer 1](images_and_label_data/writeup_images/feature_map_3.JPG)  
Layer 3 is the most abstract layer. It is almost entirely black. One important point in layer 3 is that the code is set
 to only display an image if the absolut value of the sum of the image is greater than 2. Essentially, its only 
 displaying points of the network that had any output, which is only 30 out of 400. This tells us that somehow, the
  network had determined that when these 30 points are activated in the network, and given the activation paterns within
   these layers, the image is most likely a stop sign.


