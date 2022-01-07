# Deployable SCRDET++ Object Detecion Module with Multi-GPU Processing and NITF compaitibility

This repo contains infromation for the latest SCRDET++ Python & Tensorflow-based oriented-object detector for use on arbitrarily sized single-band NITF or PNG imagery. The detector is available as a Python module (**scrdet_parallel.py**) and outputs detection results as a Pandas DataFrame. 



## Environment options for running the detector
### You have 3 options for running the module
#### 1) By downloading/cloning this repo and using the provided conda environment
#### 2) From a Docker Container available at url
#### 3) By setting up your own Python environment
        Requirements:
            - Python 3.6.10
            - tensorflow-gpu 1.13.1
            - CUDA 10.1
            - gdal 3.1.1
            - cv2 4.4.0
            - numpy 1.16.6
            - shapey 1.7.1
            - pandas 1.1.5




### Option 1) Using the provided conda environment
From the conda_environment folder on a linux terminal, create the conda environment:
```
conda env create -f scrdet_nitf_env.yml
```
Activate the conda environment:
```
conda activate scrdet_nitf_env
```
From the main folder, test that you can import the module in python
```
python
import scrdet_parallel
```

### Option 2) Using the Docker Container

1) Set up your account on container yard: url
2) Login to docker from your linux terminal: ```docker login -u username -p pasword url```
* Note: you may need to add yourself to the docker group with ```newgrp docker``` or run docker with sudo using ```sudo docker``` if you havne't been added to the docker group. 
3) Pull the image: ```docker pull url```
4) Rename the image if you want ```docker image tag url```
5) Test that you can run the docker image and import the module
```
docker run --rm -v /:/mnt/ -it scrdet_parallel ipython

from scrdet_parallel import Detector
```

## API

### The scrdet_parallel module contains the "Detector" Python class to intialize a detector instance followed by the "predict" function to run the detector instance on an image or folder of images. 

**Detector(gpu_ids,placeholder=(2048,2048,3),allow_growth=False,model='model/myproject_scrdet_Frozen.pb')**

    Initialize detector instance, assign GPUs

    Parameters
    ----------
    gpu_ids : int, [int] or csv string assigning specific GPUs for this process.
                Current version supports multiple GPUs for parallel processing
                of large images. 

    placeholder : 3D tuple (rows,cols,channels) sets sliding window size used to
                    process arbitrarily sized imagery and reserves memory on the
                    GPU's. Imagery smaller than placeholder will be zero-padded. 
                    Default is (2048,2048,3). Has been successfully tested up to
                    shape of (4096,4096,3) on a Tesla V100-DGXS-32GB GPU. 

    allow_growth : bool; Whether or not to allow other processes to allocate GPU memory
                    on the GPU's you are using. Default is False. If you are maxing
                    out GPU memeory with very large images (e.g., 4096,4096,3), you
                    will want this set to False. For futher info, see Tensorflow's
                    documentation for tf.ConfigProto gpu_options.allow_growth

    model : string, path to the tensorflow frozen graph .pb file


**predict(self,file,clips = [], conf = 0.01,
                class_list = ['class1','class2','class3'],
                virtual_mem=False, nms_iou = 0.2, h_overlap = 200,
                w_overlap = 200, max_nms_output = 200)**

    Parameters
        ----------
        file : str
            Complete path to an image, or path to a directory of images
        
        clips : list[[left1,bottom1,right1,top1]...[leftn,bottomn,rightn,topn]]
            A list of lists containing pixel values used to clip out a portion or portions of the image for 
            processing. Can be left as an empty list (i.e., clips = []) to process the whole image (this is 
            default). Must be formatted as shown. Passing a single clip would be formatted as 
            clips = [[left,bottom,right,top]]. Clips are not supported if passing in an entire directory of 
            images for detection. Pass a single image with a clips locations if you want to use clips. If clips are
            passed in, then virtual_mem will automatically be disabled (set to False).
        
        conf : float (0,1] or list of floats where each value corresponds to a specific class confidence
            Output detection if detection confidence score is greater than or equal to this value
            
        class_list : list of strings
            A name for each class
        
        virtual_mem : Bool
            Reduce memory consumption by treating NITF imagery as a virtual array, where smaller portions of the 
            NITF image are read into memory as they are needed for processing. Reduces time up front by 
            preventing the whole NITF being read into memory, but results in longer processing time overall. 
            If clips are passed in, then virtual_mem setting is overwritten and set to False.
            
        nms_iou : float; Non-Max Supression IoU threshold.

        *_overlap : int; h (height/vertical) or w (width/horizontal) pixel overlap for tiled/sliding image 
            processing window(s). Size of processing window is set by the "placeholder" parameter
            when the detector is initialized.

        max_nms_output : int; maximum number of detections to return from nms function.  

    Returns
    -------
        pd.DataFrame
            A pandas dataframe containing the following colums:
                * id - int, detection id (index of dataframe)
                * geometry - Shapely formatted polygon in NITF pixel coordinates
                * class - string, object class
                * conf - float in (0,1]
                * image_name - string
                * (class) - float, confidence score for each class, including background, as its own column
                

## Jupyter Notebook Example
### The Jupyter notebook **myproject_cv_api.ipynb** is provided with the GitLab repo for usage examples. 

## Docker Example
### omitted

## Test & Evaluate
If you ran the **myproject_cv_api.ipynb** Jupyter Notebook and have the **my_results.txt** file saved in the Test&Evaluate folder, follow the instructions in the **evaluate.ipynb** Notebook saved in the Test&Evaluate folder. The **evaluate.ipynb** will show you how to use the **objdeteval.py** object-detection evaluator module and plot results. 