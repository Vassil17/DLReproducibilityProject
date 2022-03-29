### Work so far:

Example evaluation of Taipei works, using the pretrained network. However, attempting to evaluate examples from other datasets (e.g. Rome, but also other images from Taipei), doesn't seem to work due to image size differences. Might have to resize the evaluation image to match the output of the network.

#Update: I have managed to build a PIL Image resizer, which maintains the labels - works on other example images now from the Ours Dataset (tried on Berlin and Taipei). In order to try out new images:

1. Put images in ./example_image/old_test
2. Put labels in ./example_image/labels/val/Taipei/old_val
3. Put the path to each image in ./example_image/Taipei_test.txt (IN FUTURE SHOULD AUTOMATE THIS)

Note: For step 3., you can use this command
```
find $(pwd) -maxdepth 1 -type f -not -path '*/\.*'  >> ~/TU_Delft/FCNs_Wild/example_image/Taipei_test.txt
```

4. Run 
```
>> python src/cityscapesscripts/evaluation/pil_resize.py 
```
This resizes the images to the correct (512,256) dimensions. Make sure the correct paths to the images
are set in the file.
5. Run
```
>> sh scripts/infer_city2NMD.sh 
```
This evaluates the images - the resulting visualisation can be found in  ./train_results/GACA/Taipei/iter_800/visualize