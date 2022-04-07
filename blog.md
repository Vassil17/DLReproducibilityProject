# FCNs in the Wild: Reproducibility Project for Deep Learning
## Introduction:
In this blog post we will first introduce the paper that we were tasked with reproducing, namely FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation.
We will go through the main concepts behind pixel-level image segmentation and domain adaptation. We will then explain the existing code implementations (one in PyTorch and one in TensorFlow).


## FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation
### What is Semantic Segmentation?
First let's start with a simple explanation on what the task of semantic segmentation is, as well as what network architectures are typically used to perform this task. Semantic segmentation is essentially a classification task on a pixel-level - it aims to assign a label to each pixel in a given image. In classical classification tasks, such as for example a pedestrian detector, the model typically predicts which class is present in an image. Often these classifiers only deal with predicting whether the class is in the image, without actually saying where in the image. However we often are very concerned with the location of the detected object - for example in autonomous cars, we not only care about the presence of a pedestrian, but also about where the pedestrian is within the field of view of the car. One way to solve this problem is to split the image in many small regions and then check for the presence of a pedestrian in each of them. However since the size of a pedestrian is dynamic (and also changes based on their distance to the camera), it is difficult to set one common size for this image subset. Perhaps there is a better way of detecting not only the presence of a certain object in an image, but also its location?

Semantic segmentation aims at solving this problem by classifying each pixel in an image - it takes a raw image as an input and outputs a vector with the label for each of the pixels in the original raw image. The idea behind is to not only distinguish the different objects within the image (segmentation) but also learn what they represent (semantic). For normal image classification the common approach is to use a CNN for important feature extraction and then several Fully Connected Layers as the head, which perform the classification itself. In semantic segmengation the approach is to not use fully-connected layers, but instead to use a Fully-Convolutional Network (FCN) to train the whole end-to-end process, from detecting features in the image, to classifying each of the pixels.
A very common architecture for semantic segmentation
### What is domain adaptation and why is it needed?


## Implementation
There were two implementations available for this paper: one in PyTorch (link) and one in TensorFlow (link).

### PyTorch implementation:
This implementation provides the code for obtaining the pre-trained VGG16 network and then further trains it on the GTA5 annotated dataset for semantic segmentation. However, the implementation is still work-in-progress and is missing the main parts of the paper - namely the domain adaptation adversarial training. Furthermore, the paper itself does not provide in-depth description of the hyperparameters used and the architecture. For this reason, we decided not to use this implementation and instead focused on the alternative TensorFlow implementation.

### TensorFlow implementation:
The second available implementation of the paper was written in a combination of C++ and Python using TensorFlow 1.1. The use of C++ required a compiler and since installing a compiler like cmake on Windows is a tedious process, we used a native Ubuntu installation to run the network. The implementation itself was complete and an trained model was available. This model was trained on the CityScapes dataset and then adapted on a different dataset - NMD, which also contains real-life photos of city landscape. 


### Our work:
#### Getting started:
Since the PyTorch model was largely unfinished, our team decided to use the TensorFlow implementation. Unfortunately, that implementation required a compiler, as well as some outdated and deprecated packages. Furthermore, the dependencies list only contained the MacOS versions of packages - hence we had to find the corresponding version of each package for Windows/Linux. Much of our initial work was dedicated to ensuring that all packages were compatible, as the implementation also required TensorFlow 1.1 which then requires an older version of CUDA in order to run the training process on the GPU. In the end we were able to run the implementation, however only one member of our team had a native Linux installation which made parallel work on the existing code much more difficult.

#### Data Processing:
After we set up the model, we wanted to evaluate the performance of the available trained model on some of the test images. In order to do that we had to process all the test images since they had to be in a 512x256 image size. We created an image processing function which would re-scale the images to match the input of the neural network, while also maintaining some of the label information that was built into the images themselves. A benefit of this approach was that re-scaling the images also drastically reduced the size of the training and testing datasets - from tens of GBs to only a few GB.
#### Training the model:
We adapted the provided scripts in order to use the specific datasets used in the original paper - namely the labelled SYNTHIA for training and the unlabelled Cityscapes for the domain adaptation. Our first attempt was to train the network on a Intel Core i7 CPU. The average batch processing time was around 400 seconds, which showed that training it on the CPU would be intractable. We then re-configured the network to use the laptop's RTX 2060 6GB graphics card. Unfortunately the GPU would quickly run out of memory due to the size of the model itself - lowering the batch size and training on only a subset of the training set did reduce the GPU usage but it was still not able to fit the whole model in memory. Since we could not possibly train the model on our hardware, we decided to instead evaluate the performance of the pre-trained available model on different datasets, without performing extra domain adaptation. 
