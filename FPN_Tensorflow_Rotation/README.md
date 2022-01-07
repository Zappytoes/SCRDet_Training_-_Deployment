# Modified SCRDet++ training repo provides model export script and additional training instructions. Exported model to be used with Deployable_SCRDet_Object_Detection_Module repo.

Modified SCRDet++ repo with a compatible model export script and additional training instructions. Use exported model with Deployable_SCRDet_Object_Detection_Module repo. This pipeline produces object-aligned rectangle bounding boxes and accepts quadrilateral (4 corner points) as groundtruth boxes. The original GitHub repo can be found at https://github.com/SJTU-Thinklab-Det/DOTA-DOAI/tree/master/FPN_Tensorflow_Rotation.  


## Get the pre-configured conda environment for training

From the conda_environment folder on a linux terminal, create the conda environment:
```
conda env create -f scrdet_env.yml
```
Activate the conda environment:
```
conda activate scrdet_env
```
### Alternatively, build your own environment
- python 3.5
- cuda >=10.0
- opencv (cv2)
- tfplot 0.2.0 (optional)
- tensorflow-gpu 1.13

## Pretrained Weights
You'll find pretrained weights from ImageNet in **data/pretrained_weights**. 

## Compile
```
cd libs/box_utils/cython_utils
python setup.py build_ext --inplace (or make)

cd libs/box_utils
python setup.py build_ext --inplace
```

## Formatting data for pre-processing
### Imagery
This training pipeline has been tested with PNG files, but might handle other formats as well. It's up to you to test other formats. You do NOT need to crop your data before using this pipeline, as the scripts provided here will do that work for you. For the resnet*v1d weights, use images with pixel values ranging from 0-255. 

### Groundtruth
For each image, there should be a unique ".txt" file with the same name as the corresponding image name. Each line of each text file will have the following format for each object, where each object is bounded by a quadrilateral (4 corner points) groundtruth box:
x0 y0 x1 y1 x2 y2 x3 y3 class 0

Note: the "0" at the end is just a placeholder for a "difficulty value". See the DOTA challenge website for more info. 

### A couple of example imagery and groundtruth files are provided for you in the **Sample_Training_Folder**

## Setting parameters in the **libs/configs/cfgs.py** file
At a minimum, you'll modify CLASS_NUM, NET_NAME, DATASET_NAME, VERSION, TFRECORD_PATH, ROOT_PATH (main directory for this README file), and GPU_GROUP in the cfgs.py file. NET_NAME should match the name of the network you want to use in the **data/pretrained_weights** folder. Edit any other parameters you see fit, such as IMG_MAX_LENGTH, IMG_SHORT_SIDE_LEN, PIXEL_MEAN_, and PIXEL_STD. 

## Add category info to the **libs/label_name_dict/label_dict.py** file
e.g., on line 138 of ***lable_dict.py***:
```
elif cfgs.DATASET_NAME == 'myproject':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'class1':1,
        'class2':2,
        'class3':3
    }
```

## In **data/io/read_tfrecord_multi_gpu.py**, update the ***valid_dataset*** variable to include your dataset name (must match what you have for cfgs.DATASET_NAME)
```
valid_dataset= ['myproject', 'DOTA1.5', 'ICDAR2015', 'pascal', 'coco', 'bdd100k', 'DOTA', 'HRSC2016', 'UCAS-AOD']
```

## Pre-process the data (formatting data for training)
### 1) Crop your data
Edit lines 200-223 in ***data/io/myprject/data_crop.py*** to match your dataset. Check your crop values make sense with your IMG_MAX_LENGTH and IMG_SHORT_SIDE_LEN in the cfgs.py file. Then run the script. 
```
cd data/io/myproject
python data_crop.py
```

### 2) Convert the cropped data to a tensorflow record file
```
cd data/io
python convert_data_to_tfrecord.py --VOC_dir='../../Sample_Training_Folder/TRAIN_crop_xml/'
                                   --save_dir='../tfrecord/myprject_cropped_TFrecord/'
                                   --dataset='myproject'
```

## Train the detector
```
cd tools
python multi_gpu_train.py
```
### This should create a folder named "output" in the main folder (cfgs.ROOT_PATH) containing the "summary" folder for viewing training history in Tensorboard and the "trained_weights" folder, populated based on your cfgs.SAVE_WEIGHTS_INTE setting.

## View training progress in Tensorboard
```
cd output/summary
tensorbaord --logdir=.
```

## Export the trained model and weights to a frozen graph .pb file
Edit libs/myproject_ROTATION_ARRAY_exportpb.py CKPT_PATH, OUT_DIR, PB_NAME. Then,
```
cd libs/export_pbs
python myproject_ROTATION_ARRAY_exportpb.py
```
The file ending in "_Frozen.pb" in the OUT_DIR is ready for testing or deployment. 

## Testing and model deployment
### For testing, evaluation, and deployment, please follow instructions at the SCRDET MYPROJECT PARALLEL GitLab repo: url/scrdet-myproject-parallel