from PIL import Image
import numpy as np
import os
import cv2

def resize_image(path,outPath):
        # iterate through the names of contents of the folder
        for image_path in os.listdir(path):
            # create the full input path and read the file
            input_path = os.path.join(path, image_path)
            groundTruthImg = Image.open(input_path)
            if not groundTruthImg.size == (512,256):
                print('FAIL')

def main():

   #outPath = "/home/vassil/TU_Delft/FCNs_Wild/example_image/labels/val/Taipei"
    #path = "/home/vassil/TU_Delft/FCNs_Wild/example_image/labels/val/Taipei/old_val"
    outPath = "/home/vassil/TU_Delft/Datasets/cityscapes/train/"
    path = "/home/vassil/TU_Delft/Datasets/cityscapes/train/"
    resize_image(path,outPath)



if __name__ == '__main__':
    main()