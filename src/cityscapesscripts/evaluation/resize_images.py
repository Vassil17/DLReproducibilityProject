import numpy as np
import os
import cv2

def resize_image(path,outPath):
	    # iterate through the names of contents of the folder
	    for image_path in os.listdir(path):
	    	print(image_path)
	        # create the full input path and read the file
	        input_path = os.path.join(path, image_path)
	        image_to_resize = cv2.imread(input_path)
	        if np.any(image_to_resize == None):
	        	continue
	    	else:
		        
		        # resize the image
		        resized = cv2.resize(image_to_resize, (512,256), interpolation = cv2.INTER_NEAREST)

		        # create full output path, 'example.jpg' 
		        fullpath = os.path.join(outPath, image_path)
		        cv2.imwrite(fullpath, resized)

def main():




    outPath = "/home/vassil/TU_Delft/FCNs_Wild/example_image/"
    path = "/home/vassil/TU_Delft/FCNs_Wild/example_image/old_test/"

    resize_image(path,outPath)


    
    outPath = "/home/vassil/TU_Delft/FCNs_Wild/example_image/labels/val/Taipei"
    path = "/home/vassil/TU_Delft/FCNs_Wild/example_image/labels/val/Taipei/old_val"

    resize_image(path,outPath)



if __name__ == '__main__':
    main()