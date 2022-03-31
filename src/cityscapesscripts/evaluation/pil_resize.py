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
                groundTruthImg = groundTruthImg.resize((512,256),resample=Image.NEAREST)
                groundTruthNp = np.array(groundTruthImg)
            #print(groundTruthImg.getdata()[0])

                fullpath = os.path.join(outPath, image_path)
                groundTruthImg.save(fullpath)

def main():

   #outPath = "/home/vassil/TU_Delft/FCNs_Wild/example_image/labels/val/Taipei"
    #path = "/home/vassil/TU_Delft/FCNs_Wild/example_image/labels/val/Taipei/old_val"
    outPath = "/home/vassil/TU_Delft/Datasets/cityscapes/val/"
    path = "/home/vassil/TU_Delft/Datasets/cityscapes/val/"
    resize_image(path,outPath)



if __name__ == '__main__':
    main()