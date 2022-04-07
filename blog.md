# FCNs in the Wild: Reproducibility Project for Deep Learning
## Introduction:
In this blog post we will first introduce the paper that we were tasked with reproducing, namely FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation.
We will go through the main concepts behind pixel-level image segmentation and domain adaptation. We will then explain the existing code implementations (one in PyTorch and one in TensorFlow).


## FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation


## Implementation
There were two implementations available for this paper: one in PyTorch (link) and one in TensorFlow (link).

### PyTorch implementation:
This implementation provides the code for obtaining the pre-trained VGG16 network and then further trains it on the GTA5 annotated dataset for semantic segmentation. However, the implementation is still work-in-progress and is missing the main parts of the paper - namely the domain adaptation adversarial training. Furthermore, the paper itself does not provide in-depth description of the hyperparameters used and the architecture. For this reason, we decided not to use this implementation and instead focused on the alternative TensorFlow implementation.

### TensorFlow implementation:
The second available implementation of the paper was written in a combination of C++ and Python using TensorFlow 1.1. The use of C++ required a compiler and since installing a compiler like cmake on Windows is a tedious process, we used a native Ubuntu installation to run the network. The implementation itself was complete and an trained model was available. This model was trained on the CityScapes dataset and then adapted on a different dataset - NMD, which also contains real-life photos of city landscape. 


### Our work:
Since the PyTorch model was largely unfinished, our team decided to use the TensorFlow implementation. Unfortunately, that implementation required a compiler, as well as some outdated and deprecated packages. Furthermore, the dependencies list only contained the MacOS versions of packages - hence we had to find the corresponding version of each package for Windows/Linux. Much of our initial work was dedicated to ensuring that all packages were compatible, as the implementation also required TensorFlow 1.1 which then requires an older version of CUDA in order to run the training process on the GPU.
