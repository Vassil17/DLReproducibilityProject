# FCNs in the Wild: Reproducibility Project for Deep Learning
## 1. Introduction:
In this blog post we will first introduce the paper that we were tasked with reproducing, namely FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation.
We will go through the main concepts behind pixel-level image segmentation and domain adaptation. We will then explain the existing code implementations (one in PyTorch and one in TensorFlow).

### 1.1 Presentation of paper

#### Background and main contribution
The motivation behind this paper is that fully convolutional models typically perform well in supervised settings but struggle when changing between domains, ie. if the model is trained on simulated data but must be tested in a real-data setting. The difficulty when changing between domains is that it causes shifts in the pixel-level distribution. Some specific examples of this can be changes in the light or the frequency of object appearance. For semantic segmentation, the latter is a great challenge. For this reason, it is important for learning to be able to transfer information between related settings. 

The main contribution from the article is thus a method for semantic segmentation for domain shifts. They introduce an unsupervised adversarial approach for pixel prediction for adapting to new domains. The suggested method indeed outperforms the baseline model on multiple large-scale data sets. 

#### Method
The method combines two parts for dealing with adaption. A part that deals with global changes and a part that deals with category-specific changes. The global changes relate to a shift in the marginal distribution of the feature space – this is most obvious when the domains are very different such as real and simulated data. The category-specific changes relate to differences in category-specific features such as occurrence of objects. The framework consists of a source domain with labeled images and a target domain with unlabelled images. The loss function consists of three main parts: 
* 1. A part that simply optimises the supervised segmentation in the supervised domain. The purpose is to ensure that the model does not diverge too much from the source solution and thus that the transfered information is valid. 
* 2. A part that minimises the distance between the global distributions in the two domains. This is done by adversarial learning with an alternating minimization procedure. The first objective seek to finding the parameters that will minimise the distance between the source and target domain. The second objective will train a classifier to distinguish between source and target domain and hereby estimate a distance function. The result is then that the model learns the best possible classifier and use this information to learn the parameters that can minimise this difference. 
* 3. A part for category-specific adaption by using statistics from the labeled source domain in the unlabeled target domain. For each source image containing a class c is computed the percentage of pixels whose true label is class c. The purpose is that pixels in the target domain is assigned to classes within the expected range based on the source domain. This paper has the additional contribution that it uses the lower and top 10% as well as the average value for these contraints compared to prior work that often uses just a single threshold. In that way information from a supervised setting is transferred to an unsupervised setting. 

#### Applications and Results

The method is applied to three different types of domain adaption tasks, namely between cities, between seasons as well as between synthetic and real data. To study these shifts four different datasets are applied. Cityscapes is used as target domain for all three domain shifts and also the source domain for cities-->cities.  SYNTHIA is used as source domain both for the application of season->season and synthetic -> real. GTA5 is used as source domain for synthetic-->real. BDDS is used as both source and target domain for cities-->cities. All together this represents shifts of various challenge for the model. DOUBLE CHECK THAT I GOT THIS RIGHT. 

The final results presented that we were to reproduce is the performance on adaption from synthetic to real data using GTA5 and SYNTHIA. It is clear that the proposed method outperforms the baseline model when it comes to identifying the vast majority of objects. In addition there was an ablation study to examine the effect of including the category-speficic part of the loss function. For GTA5 the category-specfic adaption offered a clear benefit but only a small improvement for SYNTHIA and cities-->cities. 

INSERT TABLE IN THIS SECTION

### 1.2 Plans for reproducibility project. NOT FINISHED
Bases on our understanding of the paper, we identified several analyses of interest. 

* Apply the method to different domain adaption tasks. The majority of the data that the article is based upon is related to cities, wether it being in different cities, in different seasons or synthetic vs. real. It could be interesting to experiment with alternative settings such as FIND EXAMPLES + DATASETS. What kind of challenge is this (large, medium, small?) 
* Including ablation study with category-speicific adaption. The effect of the CA part would be interesting for further examination, as the magnitude of improvement seemed to vary across datasets. 
* Changing constraints in CA? 
* Learning curve for different number of data samples?

## 2. FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation

### 2.1. What is Semantic Segmentation?
First let's start with a simple explanation on what the task of semantic segmentation is, as well as what network architectures are typically used to perform this task. Semantic segmentation is part of the family known as Dense Prediction Tasks and is essentially a classification task on a pixel-level - it aims to assign a label to each pixel in a given image. In classical classification tasks, such as for example a pedestrian detector, the model typically predicts which class is present in an image. Often these classifiers only deal with predicting whether the class is in the image, without actually saying where in the image. However we often are very concerned with the location of the detected object - for example in autonomous cars, we not only care about the presence of a pedestrian, but also about where the pedestrian is within the field of view of the car. One way to solve this problem is to split the image in many small regions and then check for the presence of a pedestrian in each of them. However since the size of a pedestrian is dynamic (and also changes based on their distance to the camera), it is difficult to set one common size for this image subset. Perhaps there is a better way of detecting not only the presence of a certain object in an image, but also its location?

Semantic segmentation aims at solving this problem by classifying each pixel in an image - it takes a raw image as an input and outputs a vector with the label for each of the pixels in the original raw image. The idea behind is to not only distinguish the different objects within the image (segmentation) but also learn what they represent (semantics), i.e. which class they are part of. The figure below illustrates this by colouring each pixel corresponding to a car in blue, greenery in green and pedestrains in red.

<p>
<img src="https://user-images.githubusercontent.com/69580104/162279209-3d008ac0-784f-4901-b9d2-c5eb35b43607.jpeg" width="512" height="256" />
  
[Source](https://towardsdatascience.com/semantic-segmentation-of-150-classes-of-objects-with-5-lines-of-code-7f244fa96b6c)
  
<p>

Hopefully you have seen why semantic segmentation is useful through the toy pedestrian detector example we mentioned earlier - not only does it tell you about the presence of a certain class in an image, but it shows you its exact spatial location within the image. Keeping to the current example of autonomous cars, this can be extremely powerful - your car is now not only able to detect pedestrians, but also the road, other vehicles, traffic signs and much more.
  
### 2.2. Architectures for Semantic Segmentation
Now that we've established what semantic segmentation is and why it's useful, how can you actually use machine learning to obtain such a pixel-level classification? 
  Well, for normal image classification the common approach is to use a CNN to detect and extract the important features, and then several Fully Connected Layers (FCLs) as the "head", which perform the classification task itself. One of the problems with this approach is that the FCLs require an input of fixed size - which also limits the size of the image.
In semantic segmengation the approach is to not use fully-connected layers as the end of the network, but instead to use a Fully-Convolutional Network (FCN), such as AlexNet and VGG-16, to train the whole end-to-end process, from detecting features in the image, to classifying each of the pixels. One benefit of this is that an input image of any size can be used (since the kernels or convolutions can be applied across an image of any size).
  The next figure, taken from [1], illustrates how this can be done, architecture-wise.

<p>
<img src="https://user-images.githubusercontent.com/69580104/162281381-4bdeee8a-7823-497b-a5b9-61be9ef77a51.png" width="512" height="256" />
  
[Source](https://arxiv.org/abs/1605.06211)
  
<p>

As can be seen in the figure, through pooling (subsampling) the input size is reduced throughout the network - which is beneficial because it essentially allows a pixel in a deeper layer to have a larger receptive field, i.e. to "see" more pixels from the original image. However, our task is to classify each pixel in the image, hence we need to upsample, which is exactly what happens at the final "pixelwise prediction" layer. Now a pixel-wide prediction can be output at the last layer, which means that we can compare it to some ground truth and see how well the network performs. A commonly used loss function in this case is the *Cross-Entropy Loss*, which is the same as the one used for classical image classification tasks, only applied to each pair of pixels between output and ground truth. One issue is that the predictions might be too "coarse" due to the subsampling within the network and then upsampling to match the input image size. There are many different ways of dealing with this, for example by including skip layer connections which pass some shallow layers and fuse them with the coarse deep layers to obtain a better prediction. 
  
### 2.3. What is domain adaptation and why is it needed?


## 3. Implementation
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
  
## 4. Perspective to Edward Raff 2019 'A Step Toward Quantifying Independently Reproducible Machine Learning Research'. 
Reproducibility was not succesful in this case and in the above we have evaluated our own challenges in this process. We will end the blog with some perspectives to the seminar paper from Week 2; 'A Step Toward Quantifying Independently Reproducible Machine Learning Research' by Edward Raff, and evaluate our paper 'FCNs in the Wild' based on a selection of their findings. 
