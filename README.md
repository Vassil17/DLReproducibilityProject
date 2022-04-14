# FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation Implemented by Tensorflow
Paper link: [https://arxiv.org/abs/1612.02649](https://arxiv.org/abs/1612.02649)


## Intro 
Tensorflow implementation of the paper for adapting semantic segmentation from the (A) Synthia dataset to Cityscapes dataset and (B) Cityscapes dataset to Our dataset.

## Installation
* Use Tensorflow version-1.1.0 with Python2
* Build ccnn

	```
	cd fcns-wild
	mkdir build
	cd build
	cmake ..
	make -j8
	```

## Dataset

* Download [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
* Download [Synthia Dataset](http://synthia-dataset.net/downloads/)
	* download the subset "SYNTHIA-RAND-CITYSCAPES" 
* Download [NMD Dataset](https://yihsinchen.github.io/segmentation_adaptation/#Dataset)
	* contains four subsets --- Taipei, Tokyo, Roma, Rio --- used as target domain (only testing data has annotations) 
* Change the data path in files under folder "./data"
## Testing
* Download and testing the trained model 

	```	
	cd fcns-wild
	sh scripts/download_demo.sh
	sh scripts/infer_city2NMD.sh 	# This shell NMD is using Taipei
	```

	The demo model is cityscapes-to-Taipei, and results will be saved in the `./train_results/` folder. Also, it shows evaluated performance. (the evaluation code is provided by Cityscapes-dataset).

If you would like to run some testing runs you need to have all test images in ./example_image and all corresponding labels in ./example_image/labels/val/Taipei/ . Make sure to have a Taipei_test.txt file inside the ./example_image folder which contains the file location of each image - some bash scripts that easily automate this are the following:

From the folder containing the images run (saving the Taipei_test.txt file to the ./example_image folder):
```
>> find $(pwd) -maxdepth 1 -type f -not -path '*/\.*' | sort  >> ./Taipei_test.txt
```

## Blog Post:
Our blog can be found in the blog.md file, showing our work on the reproducibility project plus the results we obtained. Some of the figures and images had to be lowered in resolution in order to be converted to PDF, so for the full resolution blog post please check the blog.md file.

