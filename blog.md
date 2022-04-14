# FCNs in the Wild: Reproducibility Project for Deep Learning
## 1. Introduction:
In this blog post we will first introduce and go through the main concepts behind pixel-level image segmentation and domain adaptation. Then we will introduce the paper that we were tasked with reproducing, namely FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation. Finally, we will then explain the existing code implementations (one in PyTorch and one in TensorFlow), our own work and the results we obtained.


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

First of all, let's compare these two images:
  
<p>
<img src="https://user-images.githubusercontent.com/69580104/162791518-5af5c7f8-82d8-48ce-bc03-ff95ca9c9e07.png" width=50% />
  
[Source: TripSavvy]
  
<p>
<p> <img src="https://user-images.githubusercontent.com/69580104/162792068-e62c59a2-3b7a-4f3f-8b2a-24245874a049.png" width=50%/>
  
[Source: NYCgo.com]
<p

For a human it is relatively trivial to realise that they are both photos of a square, containing a lot of signs, people, and some vehicles. There are stark differences between the way the images look in terms of colour and intensity, but a human can attribute all that to the fact that the top image is obviously taken on a snowy winter day. Unfortunately, neural networks often cannot achieve this reasoning - it is common knowledge that models will cheat or find a shortcut, if one is available. This, in turn, results in such models not being very good at generalising to new environments, especially ones that look so different from what they are used to - even if they semantically contain the same information.
   
 Hopefully this little example has illustrated how vastly different images of semantically the same objects can look. It is not just lightning and weather conditions that can affect the way an instance of a class appears - in winter people dress very differently from summer, yet they are still people that should be recognised as such. Then this begs the question -How can we then teach a neural network to associate such instances with the ones it knows? Well, this is precisely what domain adaptation aims to solve.
   
More often than not it is not the lack of data that is the problem, but the lack of labels - as the process of annotation is both time-consuming and expensive, especially when one is dealing with a more complex task like pixel-wise segmentation and classification. As such, if we want to deploy our semantic segmentation network into a new city, we will often know what this city looks like and already have images of it - and if not, the collection of such raw data is not particularly difficult. One of the ideas of domain adaptation is exploiting exactly this - how can we teach our network to adapt its acquired knowledge to this new environment, without requiring the use of expensive labelled data? Well this is what the paper *FCNs in the Wild* is about, and we will introduce that in the next section.
   
### 2.4 The paper

#### Background and main contribution
The motivation behind this paper is that fully convolutional models typically perform well in supervised settings but struggle when changing between domains, ie. if the model is trained on simulated data but must be tested in a real-data setting. The difficulty when changing between domains is that it causes shifts in the pixel-level distribution. Some specific examples of this can be changes in the light or the frequency of object appearance. For semantic segmentation, the latter is a great challenge. For this reason, it is important for learning to be able to transfer information between related settings. 

The main contribution from the article is thus a method for semantic segmentation for domain shifts. They introduce an unsupervised adversarial approach for pixel prediction for adapting to new domains. The suggested method indeed outperforms the baseline model on multiple large-scale data sets. 

#### Datasets
There were three datasets that were used in the paper for various domain adaptations: Cityscapes, SYNTHIA and GTA5. Starting with Cityscapes, which contained 5000 images from several cities which were used on a 59.5/30.5/10 train/validation/testing split. Secondly, the SYNTHIA dataset used contained 9000 synthetic city images which had Cityscape-compatible annotations. Lastly, GTA5 which contained 24,966 labeled images from the video game Grand Theft Auto 5 and gathered a subset of these images whose labels were compatible with the Cityscapes. 
   
#### Method

   **Baseline model**
 The paper uses a dilated fully convolutional network VGGNet with 16 layers as the baseline. Dilated convolution means that the size of kernel is expanded by inserting empty spaces so that some pixels are skipped when performin the convolution operation. This way more of the receptive field is covered without the need for learning more parameters. After the last convolutional layers they make use of bilinear up sampling to produce segmentation in same resolution as the input. Below you can see the general architecture of the full model, as illustrated in the original paper. We will explain how the domain adaptation is achieved in the next section.

<p>
<img src="https://user-images.githubusercontent.com/69580104/163430418-0a4b89bb-9c72-4c0d-8d13-0c36d56c54d2.png" width="400" height="256" />
  
[Source](https://arxiv.org/pdf/1612.02649.pdf)
  
<p>
**Domain adaption**   
The method combines two parts for dealing with adaption. A part that deals with global changes and a part that deals with category-specific changes. The global changes relate to a shift in the marginal distribution of the feature space – this is most obvious when the domains are very different such as real and simulated data. The category-specific changes relate to differences in category-specific features such as occurrence of objects. The framework consists of a source domain with labeled images and a target domain with unlabelled images. The loss function consists of three main parts: 
   
   ![image](https://user-images.githubusercontent.com/101123359/162801773-62e38175-9bed-4a28-950e-9c84b14e85eb.png)
   
* 1. _Lseg_ simply optimises the supervised segmentation in the supervised domain, thus mapping the source images _Is_ to the source labels _Ls_ The purpose is to ensure that the model does not diverge too much from the source solution and that the transfered information is actually fitting. 
* 2. _Lda_ minimises the distance between the global distributions in the two domains. This is done by adversarial learning with an alternating minimization procedure as shown below: 
   
   ![image](https://user-images.githubusercontent.com/101123359/162803632-bc050220-12dc-47b9-bb83-0c06f92e77da.png)

The first objective seek to finding the parameters that will minimise the distance between the source and target domain. The second objective will train a classifier to distinguish between source and target domain and hereby estimate a distance function. The result is then that the model learns the best possible classifier and use this information to learn the parameters that can minimise this difference. 
   
* 3. _Lmi_ is the category-specific adaption using images from the target domain _It_ and label statistics transferred from the source domain. For each source image containing a class c the percentage of pixels whose true label is class c is computed. The purpose is that pixels in the target domain is assigned to classes within the expected range based on the source domain. This paper has the additional contribution that it uses the lower and top 10% as well as the average value for these contraints compared to prior work that often uses just a single threshold. For the predictions it is also computed the percentage of pixels that are assigned to each class, and a label is assigned if the model has predicted at least 10% of what is expected for that class. In that way information from a supervised setting is transferred to an unsupervised setting. 
  
   
#### Applications and Results
The paper presented results with various adaptations, ranging from mild to more drastic differences between the two domains. The method was applied to three different types of domain adaption tasks, namely between cities (small shift), between seasons as well as between synthetic and real data (large shift). Cityscapes is used as target domain for all three domain shifts and also the source domain for cities-->cities.  SYNTHIA is used as source domain both for the application of season->season and synthetic -> real. GTA5 is used as source domain for synthetic-->real. BDDS is used as both source and target domain for cities-->cities. All together this represents shifts of various challenge for the model. 

Below we have posted two tables containing results from the paper for the shifts aforementioned, the first one corresponds to the large shift, using GTA5 and SYNTHIA, while the second corresponds to the small shift, only using Cityscapes. Note how there are three rows per experiment, the first row is the benchmark while the bottom two rows are the results from the method in the paper that has been split in two for ablation purposes to examine the effect of including the category-speficic part of the loss function vs. only including lobal changes in the loss function. 
 
<p style="text-align: center;">Table 1: Large domain shift, trained on videogame/syntethic cities tested on real cities</p>

<p>
<img src="https://i.imgur.com/ZjODpjx.png" width="641" height="218" />
  
[Source](https://arxiv.org/pdf/1612.02649.pdf)
  
<p>
  
Table 1 shows how the network behaved when being trained on videogame/synthetic scenery then tested with real cities, as one can see some objects were more adaptable than others such as buildings and roads while recognizing other objects showed no carry over (like trains). It is clear that the proposed method outperforms the baseline model when it comes to identifying the vast majority of objects. For GTA5 the category-specfic adaption offered a clear benefit but only a small improvement for SYNTHIA. 
  
<p style="text-align: center;">Table 2: Small domain shift, trained on real cities tested on different real cities</p>
<p>
<img src="https://i.imgur.com/rWHatBT.png" width="639" height="109" />
  
[Source](https://arxiv.org/pdf/1612.02649.pdf)
  
<p>
  
Table 2 (shown above) corresponds to the small shift in domain. Note how, compared to the previous the table, the network shows high adaptability which is intuitive as the change is not as drastic. We present this table as when performing our own experiments we will use this for comparison given in our experiments we looked at adaptability between different cities. 

The image below shows an example of a medium domain shift across seasons. There are three seasons available, Summer, Fall and Winter allowing for 6 shifts. For 12 out of 13 objects the proposed method yields better results than the VGGNet base model. The image shows a shift from Fall to Winter, and it is noted how the roads are made white after the adaption to simulate snow, whereas the cars have same appearance as in the fall. 
  
  ![image](https://user-images.githubusercontent.com/101123359/162815443-85a9f5fb-5fa4-4e6f-8ffd-0b4d182b0938.png)

  
  
## 3. Plans for reproducibility project.
The results presented above is what we were to reproduce, namely the performance on adaption from synthetic to real data using GTA5 and SYNTHIA.
Bases on our understanding of the paper, we identified several analyses of interest. 

* Apply the method to different domain adaption tasks. The majority of the data that the article is based upon is related to cities, wether it being in different cities, in different seasons or synthetic vs. real. Another interesting application within computer vision is medical images, where unlabeled data often exceeds labeled data, and where the data distribution can vary because of differenct devices obtained the images, or the procedure of taking the images itself. The article _Adaptive adversarial neural networks for the analysis of lossy and domain-shifted datasets of medical images_ https://www.nature.com/articles/s41551-021-00733-w  was inspiration for this kind of analysis, data in form of images of embroyes among other things are available here https://osf.io/3kc2d/.
  
* Closer examination of the effect of category-speicific adaption (CA). The magnitude of improvement seemed to vary across datasets when looking at the results table, and further studies about the actual gain of including CA would have been interesting, for example by using different sample sizes or data augmentation of the available data.  

* Lastly, the model is trained and evaluated on large amount of data, motivating an analysis of the behaviour of the learning curve ie. the gain in performance when including more samples.


## 4. Implementation
There were two implementations available for this paper: one in PyTorch https://github.com/Wanger-SJTU/FCN-in-the-wild and one in TensorFlow https://github.com/stu92054/Domain-adaptation-on-segmentation/tree/master/FCNs_Wild , neither from the authors themselves.

### PyTorch implementation:
This implementation provides the code for obtaining the pre-trained VGG16 network and then further trains it on the GTA5 annotated dataset for semantic segmentation. However, the implementation is still work-in-progress and is missing the main parts of the paper - namely the domain adaptation adversarial training. Furthermore, the paper itself does not provide in-depth description of the hyperparameters used and the architecture. For this reason, we decided not to use this implementation and instead focused on the alternative TensorFlow implementation as recommended by our TAs. 

### TensorFlow implementation:
The second available implementation of the paper was written in a combination of C++ and Python using TensorFlow 1.1. The use of C++ required a compiler and since installing a compiler like cmake on Windows is a tedious process, we used a native Ubuntu installation to run the network. The implementation itself was complete and an trained model was available. This model was trained on the CityScapes dataset and then adapted on a different dataset - NMD, which also contains real-life photos of city landscape. 


### Our work:
  
#### Getting started:
Since the PyTorch model was largely unfinished, our team decided to use the TensorFlow implementation. Unfortunately, that implementation required a compiler, as well as some outdated and deprecated packages. Furthermore, the dependencies list only contained the MacOS versions of packages - hence we had to find the corresponding version of each package for Windows/Linux. Much of our initial work was dedicated to ensuring that all packages were compatible, as the implementation also required TensorFlow 1.1 which then requires an older version of CUDA in order to run the training process on the GPU. In the end we were able to run the implementation, however only one member of our team had a native Linux installation which made parallel work on the existing code much more difficult.

#### Datasets
The tensorflow implementation made use of the datasets from the paper (Cityscapes, SYNTHIA, GTA5) and additionally used a database "NMD Database" they had developed which was made of images from Taipei, Tokyo, Roma & Rio. Table 3 shows some more detail about the NMD Dataset.
<p style="text-align: center;">Table 3: NMD Dataset features</p>
<p>
<img src="https://i.imgur.com/ksPBiZ0.png" width="641" height="118" />
  
[Source](https://yihsinchen.github.io/segmentation_adaptation/#Dataset)
  
<p>

#### Data Processing:
After we set up the model, we wanted to evaluate the performance of the available trained model on some of the test images. In order to do that we had to process all the test images since they had to be in a 512x256 image size. We created an image processing function which would re-scale the images to match the input of the neural network, while also maintaining some of the label information that was built into the images themselves. A benefit of this approach was that re-scaling the images also drastically reduced the size of the training and testing datasets - from tens of GBs to only a few GB.
  
#### Training the model:
We adapted the provided scripts in order to use the specific datasets used in the original paper - namely the labelled SYNTHIA for training and the unlabelled Cityscapes for the domain adaptation. Our first attempt was to train the network on a Intel Core i7 CPU. The average training time for a single batch was around 400 seconds, which showed that training it on a CPU would be intractable. We then re-configured the network to use the laptop's RTX 2060 6GB graphics card. Unfortunately the GPU would quickly run out of memory due to the size of the model itself - lowering the batch size and training on only a subset of the training set did reduce the GPU usage but it was still not able to fit the whole model in memory. The weights of the pre-trained VGG-16 model itself are around 3 GBs in size, which is nearly half the memory of the RTX 2060 graphics card. We considered using a different pre-trained model as the base, but found that the Tensorflow implementation was built around the VGG-16 and it was not possible to simply switch for a lighter model.
  Since we could not possibly train the model on our hardware, we decided to instead evaluate the performance of the pre-trained available model (trained on Cityscapes and adapted to Taipei from the NMD dataset) on different datasets, without performing further domain adaptation. 
  
#### Results:
Since we were not able to train the model, we decided to instead test how well it generalises to cities that it has not seen before. While the idea of the model is to transfer the learning to a new domain using by training on some unlabelled data from the target domain, it can still be worthwile to see if a model trained on source A and adapted to target B can perform well on a new data set C. While the best performance will be certainly obtained by first adapting the network to the new city, this process requires a complete re-training of the model, which as we saw is computationally expensive, both in terms of GPU memory requirements, and also possibly in terms of training time. Unfortunately the authors did not disclose the hardware that they performed the training on, as well as the training time that it took, so we have to base our assumptions on the observations from trying to train the model on our hardware. 

We tested the Cityscapes-to-Taipei model on three other cities from the NMD dataset - Tokyo, Rio and Rome. All four cities are quite different from each other, both geographically and architecturally, which could show some interesting results regarding the generalisation capabilities of the model. Furthermore, we tested the performance on some of the data from the Berkeley dataset (that the original paper adapted to), however, due to differences in the dataset labelling format we were only able to obtain qualitative results (in the form of images) and no IoU data.

  
First let's qualitatively analyse some of the images from Taipei, the city that the model was adapted to:
  
![image](https://user-images.githubusercontent.com/69580104/162591383-6f0e5e52-6a5c-4f4b-8088-9f3869358203.png)
  
In general the model does not perform very well, even on the dataset from the city that it was adapted to. In the first image some of the street signs are segmented very well, while others are not detected at all. On the other hand, motorbikes and cars are qualitatively segmented well in most cases.
  
![image](https://user-images.githubusercontent.com/69580104/162591389-e399d5b4-1601-4740-a8d4-540f39a24b96.png)
  
In this image much better qualitative results can be observed - the pedestrians' bodies are well segmented, as well as the folliage, street sign and motorbikes. There are also much fewer noisy detections than in the first image.
  
![image](https://user-images.githubusercontent.com/69580104/162591395-ca91f2ca-b543-4a18-bdb4-0a4f1321487b.png)
  
The final that we're showing also shows good detection of vehicles, but it the model seems to have difficulty differentiating between pedestrian and motorcyclist. Furthermore some noise can again be seen around the image - mainly parts of the building that have been segmented as folliage.
  

Now we'll look at a few images from Rome, which is another one of the test cities in the NMD dataset.  
  
![image](https://user-images.githubusercontent.com/69580104/162591189-a3e3d651-c7ba-47ad-9f2e-6e73a3821837.png)
![image](https://user-images.githubusercontent.com/69580104/162591213-1d890297-7c2e-416c-9a93-c82b9c4cd45e.png)
![image](https://user-images.githubusercontent.com/69580104/162591304-cb77b60b-62f6-4e6e-878d-5b2cf3b3a8bc.png)
  
It seems that the model is quite good at segmenting cars and folliage, however, that was a noticeable lack of pedestrians in the test images for this dataset. The segmentation of pedestrians in the few images that do contain them seems to be quite coarse and also sometimes misclassifies pedestrian and motorcycle.
 
In the Rio dataset there was more pedestrians, but it struggled more with surface detection.
  
![image](https://user-images.githubusercontent.com/69580104/162591742-3077228b-3cc4-4d1b-9e61-44a9431fe750.png)
![image](https://user-images.githubusercontent.com/69580104/162591751-b26fed72-4f85-4f53-9f27-dd4445caad37.png)
![image](https://user-images.githubusercontent.com/69580104/162591762-27e136f1-af8d-487f-8f48-f1240ef205dc.png)
  
It mis-classified the a large majority of the racks of bikes, and constantly struggles with differentiating between sidewalks and roads. It really aggressively identified vegetation however, even to the point of detecting it in places where there was none.
  
Tokyo exhibited a different issue.

 ![pano_00022_2_0](https://user-images.githubusercontent.com/51253916/163423804-1ee00aa4-ee15-41c4-b0ca-85184696192c.png)
 ![pano_00320_0_180](https://user-images.githubusercontent.com/51253916/163423834-4ffc8e2c-97f2-40d1-94ff-90a36c69ecd1.png)
![pano_01364_2_0](https://user-images.githubusercontent.com/51253916/163423905-fd741aa8-c9a3-4ec5-8c41-11a95bb27b69.png)

The model over identified pedestrians in the images, and even confused traffic cones for flamboyantly dressed citizens. Most of the images also had many small blobs of mis-identified areas all across the image. It could be the that the domain was different enough to confuse the model.
  
Berkley's main difference between the other datasets was the prevalence of a car hood in most of the images and dark or rainy images. Below are some more consistent examples.
  
 ![7dc08598-f42e2015](https://user-images.githubusercontent.com/51253916/162824355-6845dd9f-bc40-48eb-a25f-628603b78982.png)
 ![7d15b18b-1e0d6e3f](https://user-images.githubusercontent.com/51253916/162824405-c4b200cb-dfc6-404f-8b07-e187d49f5d10.png)
![7daa6479-67988f3f](https://user-images.githubusercontent.com/51253916/162824543-bbae2c4c-3f2e-4954-b7c6-a90ffd2b2c5f.png)

The classification in this dataset is very amorphous, with little fine detail to speak of and lots of mis-classification of objects as foliage. The dataset also lacked a lot of pedestrians and struggled in dealing with the car hood in the bottom part of the frame.
  
<img src="https://user-images.githubusercontent.com/51253916/163428392-b65e8db2-9b4c-4609-b197-e79512796567.png" width="639" height="175" />

## 5. Perspective to Edward Raff 2019 'A Step Toward Quantifying Independently Reproducible Machine Learning Research'. 
Reproducibility was not succesful in this case and in the above we have evaluated our own challenges in this process. Although the main reason for lack of reproducibility was the GPU requirements for training the model, we also find it relevant to evaluate the article _FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation_ against the seminar paper from Week 2; 'A Step Toward Quantifying Independently Reproducible Machine Learning Research' by Edward Raff. _Readability_ was shown to be the most significant feature in terms of reproduction, and in our subjective evaluation of _FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation_ there are room for improvement in terms of better language use. The information given is very comprehensive and included in some long sentences. The paper required multiple read-throughs to get a good sense of the methodology. Lastly we can relate to Edward Raff's suggestions for non-reproducibility, there were some missing details in terms of explaning the algorithm, particularly the discriminator function where information about the distance measure for the global adaption, and additionally no information was given about the hyperparameters. In conclusion, besides the non-triviality of running the code, this was indeed a challenging paper for reproduction and we hope that future groups will have more luck!  
  
## 6. Individual work 
  
Ane Cathrine: 
  * Work on implementation: Attempt on PyTorch code, attempt on running tensorflow through Colab, Cmake gui, linux compiler cygwin64, Ubuntu terminal environment with Windows Subsystem for Linux. 
  * Work on blog post: Section 2.4, Section 3, Section 5. 

Joseph Krueger:
  * Worked on getting Pytorch implementation finished, attempted at running Tensorflow on a virtual machine but ran out of vRAM.
  * Picked and analyzed results for the results section of the blog.
  * Designed and printed project poster
  
Vassil Atanassov:
  * Work on Tensorflow implementation: Created a tool to rescale the dataset images, attempted to re-train (and adapt) the model; tune model parameters to attempt to lower the GPU costs (unsuccesful). Ran the pre-trained model on the different NMD dataset cities and Berkeley dataset, and collected results (quantitative and qualitative).
  * Work on blog post: Sections 2.1, 2.2, 2.3, 4
