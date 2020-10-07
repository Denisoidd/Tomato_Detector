# Tomato Detector
## How to launch train process or test
The best and the easiest way is to use Google Colab notebook to run the implemented code. To do so, open the `experiments.ipynb` file in Google Colab and follow the instructions inside.

## Project
This project is a binary classification problem with small and unbalanced dataset. In other words as an input we have an image of food and as output we need to say if this food contains
any traces or whole tomatoes. 

In order to solve that problem deep CNN were used. The project could be divided in the following parts:
1. Data extraction, preparation and augmentation
2. Training process
3. Discriminative Localization
Let's talk more precisely about these 3 parts !

### Data extraction, preparation and augmenation
The first part was to extract usefull images that contain some traces of tomatoes. For each image we had `json` file with annotations of which types of products it contains and also we
had a `csv` file with decoding these annotation names into understandable for human products. So we've taken all `keys` which contain `tomato` or `Tomato` inside and extracted those
images. At the end we had `542` images that contains tomatoes and `2458` images without them.

The dataset is unbalanced and to deal with that problem there are many different approaches. First one is to downsample the bigger class, but we've chosen to use **data augmenetation** 
technics to deal with that problem. So we've applied random flip, random rotation and random zoom.

Also images were resized from `600 x 600` to `256 x 256` cause we don't really need a so big resolution for binary classification. Of course if it was object detection problem it would be much more important to have a bigger resolution. 

Just after data augmentation layers **rescaling** operation was applied in order to uniform the input into Neural Network.

The data were splitted by `80%` for train and `20%` for validation.

### Training process
Neural Network was mostly based on Convolution Layers with growing number of filters. Also in order to reduce the size of features we've used Max Pooling layers. At the end of the 
network instead of commonly used Fully Connected Layer we've decided to test **GAP** (Global Average Pooling) which take the average of each feature map before it. And at the end we had
one Fully Connected Neuron with **sigmoid** activation to express the probability. After each Convolution Layer we've applied **ReLU** activation function.

Number of filters in each Convolution Layer: `[16, 32, 64, 128, 256]`

Loss function is a Cross Entropy loss for two classes (with tomato or without).

As an optimizer we were using Adam optimizer.

Also the in training process we were using **Accuracy**, **Precision** and **Recall** metrics. However it was not possible to load saved model in Keras correctly with two last metrics so **Precision** and **Recall** were abandoned. 

### Discriminative localization

Once we've trained the network we can try to predict the approximate position of tomato in the image. In order to that directly without retraining the model for object detection
task we can do the following. Firstly we will extract the output of the `last convolution layer` (we will have `256` different filters). Then in order to combine them in something 
meaningful we will multiply them by the coefficients of the last `dense layer` (it has `256` input weights) and finally we will take a sum over these 256 filters. 

This combination will give us approximate positions of detected tomatoes as it was shown in [article](https://arxiv.org/pdf/1512.04150.pdf).

## Results

Talk about loss results of training and accuracy. Put graph here.

Put some images of discriminative localization.

## Future work

Learning with few image data is a challenging task and there are many different approaches that could be tested. For example, it would be very interesting to use transfer learning 
technics in order to increase the accuracy of the algorithm. We can use pretrained models on ImageNet which have near 1000 different classes and then finetune it on our small 
dataset. I'm sure that this approach could perfom even better than ours because it will use the information from millions images. It would be interesting to test lightweight backbones such as EfficienNetB0 and MobilenetV2 / V3 to see its performances on Binary Classification tasks.





