# small library to evalute text file output of object detection text files

#import os
import numpy as np
from shapely.geometry import Polygon
#import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

# define intersection over union
def iou(det_bbox,gt_bbox):

  # convert groundtruth coordinates to list of tuples
    pts = []
    for step in np.arange(0,len(gt_bbox),2):
        pts.append(tuple(gt_bbox[step:step+2]))

    gt_poly = Polygon(pts) # create gt polygon
    
   # convert detection box coords to list of tuples
    pts = []
    for step in np.arange(0,len(det_bbox),2):
        pts.append(tuple(det_bbox[step:step+2]))
        
    det_poly = Polygon(pts) # create detection polygon
    
    intexu = det_poly.intersection(gt_poly).area/det_poly.union(gt_poly).area

    return intexu
# End intersection over union

# Evaluation code
def Eval(gt_file, det_file, iou_thresh, class_list, score_thresh=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
    
    '''
    Evaluates a detection file against a groundtruth file for object-oriented quadrilaters.
    Returns precision, recall, F1-Score, True-Positives, False-Positives, False-Negatives,
    and a confusion matrix
    
    Usage: p,r,f1,tp,fp,fn,confm = Eval(gt_file, det_file, iou_thresh, score_thresh, class_list)
    
    gt_file : str; path to single groundtruth text file containing ALL annotations for ALL images
              you are evaluating for. Each line of the groundtruth text-file is formatted
              as follows:
                  image_id class_name x0 y0 x1 y1 x2 y2 x3 y3
                  
    det_file : str; path to single detection text file containing ALL detection info for ALL images
              you tested on. Each line of the detection text-file is formatted
              as follows:
                  class_name image_id confidence_score x0 y0 x1 y1 x2 y2 x3 y3 score_class_0:n
                  
                  Note: Including the score for each class (score_class_0:n) is not required
    
    iou_thresh : float; The acceptable Intersection-Over-Union for a detection to be considered a 
                 True-Positive
    
    score_thresh : iterable : a list or array of confidence scores to evaluate against. E.g., 
                   score_thresh = np.arange(0.0,1.1,0.1)
                   
    class_list : list; An list of strings corresponding to the classes in the groundtruth (does
                 not include the 'background' class). E.g., class_list=['class1','class2','class3']
                 
    Returns : 
        
    '''

    # 1) First read the formatted text files and turn the info into lists or numpy arrays
    #gt_file = '/content/drive/My Drive/Colab Notebooks/DOTA-DOAI-master/FPN_Tensorflow/tools/xView18_Val_0_gt/labels_gt_new.txt'
    #det_file = '/content/drive/My Drive/Colab Notebooks/DOTA-DOAI-master/FPN_Tensorflow/tools/test_xView18_300004_rot_0.3NMS/JohnsNet_v1/MASTER_Val_0_test_xView18_300004_steps_dt.txt'
    
    # 1) read gt file
    with open(gt_file, 'r') as f:
        gt_content = f.readlines()
    gt_content = [x.strip() for x in gt_content]
    f.close()
    
    # 2) read detection file
    with open(det_file, 'r') as f:
        det_content = f.readlines()
    det_content = [x.strip() for x in det_content]
    f.close()
    
    # 3) convert everything to arrays
    gt_bbox = np.empty((len(gt_content),8),dtype=np.float)
    gt_im_id = []
    gt_class = []
    for entry in np.arange(0,len(gt_content)):
      temp = gt_content[entry].split(' ')
      #print(temp), print(temp[0],temp[1])
      gt_im_id.append(temp[0])
      gt_class.append(temp[1])
      gt_bbox[entry,:] = np.asarray([temp[2],temp[3],temp[4],temp[5], temp[6],temp[7], temp[8],temp[9]],dtype=np.float)
      #print(gt_bbox[entry])
    gt_im_id = np.asarray(gt_im_id,dtype=str)
    gt_class = np.asarray(gt_class,dtype=str)
    
    det_bbox = np.empty((len(det_content),8),dtype=np.float)
    det_im_id = []
    det_class = []
    det_score = []
    for entry in np.arange(0,len(det_content)):
      temp = det_content[entry].split(' ')
      det_class.append(temp[0])
      det_im_id.append(temp[1])
      det_score.append(temp[2])
    
      # xy = np.asarray(temp[3:],dtype=float)
      # xs = xy[np.array([0,2,4,6])]
      # ys = xy[np.array([1,3,5,7])]
      # det_bbox[entry,:] = np.asarray([np.min(xs),np.min(ys),np.max(xs),np.max(ys)])
    
      #det_bbox[entry,:] = np.asarray(temp[3:])
      det_bbox[entry,:] = np.asarray(temp[3:11])
    
    # create master copies of the groundtruth
    gt_bbox_master = np.copy(gt_bbox)
    gt_im_id_master = np.copy(gt_im_id)
    gt_class_master = np.copy(gt_class)
    
    print('converted txt files to arrays')
    ### end turn text file info into numpy arrays or lists
    
    # 2)  do the eval with confidence score thresholding
    #iou_thresh = 0.1
    #score_thresh = np.arange(0.0,1,0.1)
    
    
    confm = np.zeros((len(class_list)+1,len(class_list),len(score_thresh))) # confusion matrix 
    tp = np.zeros((len(score_thresh),len(class_list))) # true positives matrix
    fp = np.zeros((len(score_thresh),len(class_list))) # false positives matrix
    fn_1d = np.zeros(len(class_list)) # false negatives vector
    # Count the possible false positives (we begain assuming all 
    # groundtruth will be missed (i.e., false negative) and then prove otherwise during the eval)
    for entry in np.arange(0,len(gt_class_master)):
      index = np.where(np.array(class_list) == gt_class_master[entry])[0][0]
      fn_1d[index] = fn_1d[index] + 1
    fn = np.ones((len(score_thresh),len(class_list)))*fn_1d # convert the vector to a matrix
    
    # test for each score threshold
    #for st in tqdm_notebook(np.arange(0,len(score_thresh))): # the tqdm is just for the completion bar
    for st in np.arange(0,len(score_thresh)): # the tqdm is just for the completion bar
      #print(st)
    
      # reset the groundtruth arrays
      gt_bbox = np.copy(gt_bbox_master)
      gt_im_id = np.copy(gt_im_id_master)
      gt_class = np.copy(gt_class_master)
    
      # Loop over each detection and determine if it's a tp or fp
      #pbar = tqdm_notebook(np.arange(0,len(det_content)))
      for det_num in np.arange(0,len(det_content)):
    
        # track the index of the class (0-17)
        class_index = np.where(np.array(class_list) == det_class[det_num])[0][0]
    
        # first, do the score thresholding
        # print('score=',det_score[det_num])
        if np.float(det_score[det_num]) < score_thresh[st]: # if this detection score is less than the 
                                                            # current score threshold, we ignore it
          pass
    
        else: # otherwise, determine if this detection is a TP or FP
    
          # # track the index of the class (0-17)
          # class_index = np.where(np.array(class_list) == det_class[det_num])[0][0]
          #print(det_class[det_num], 'is position', class_index)
    
          # 1) find where the detection image id matches the gt image id's
          ind_im_id = np.where(gt_im_id == det_im_id[det_num])[0] # these are the indices where the image_id's match
    
          if len(ind_im_id) == 0: # if there are no image id matches in the groundtruth, you have either already used all the groundtruth, or you have images id's in your detection data that are not in your groundtruth labels. A detection at this stage must be a mislabled "background"
            #print('no image match for det num ',det_num)
            #print('WARNING!!!!!!!! You have detections on images that are not in the test set!!!!')
            fp[st,class_index] = fp[st,class_index] + 1
            confm[0,class_index,st] = confm[0,class_index,st] + 1 # add 1 to the background target for this selected class and score thresh
            
    
          else: # if there are image id matches (like their should be)...
            #print('det#',det_num,' matching image found')
            #print(ind_im_id)
            #print(gt_im_id[ind_im_id])
    
            # 2) find indices where the indices of matching image id's also match the class
            ind_of_ind = np.where(gt_class[ind_im_id] == det_class[det_num])[0] # these are the "indices-of-the-indices" that match the class
    
            if len(ind_of_ind) == 0: # if the detection class doesn't match any classes in the image, it's a false positive
              #print('no class match for det num ',det_num)
              fp[st,class_index] = fp[st,class_index] + 1 # increase the false positive count by 1 at the corresponding score-thresh and class element 
               
              # here we can figure out what the detection was misidentified as by finding the gt with the highest iou with this detection
              my_ious = []
              # 3) find the IOU's for all the NON-matching groundtruths
              for my_image_class_bboxes in gt_bbox[ind_im_id]: # loop over each possible TP and find the IOU
                my_ious.append(iou(det_bbox[det_num],my_image_class_bboxes)) # append the IOU vlaue to this list
    
              my_ious = np.asarray(my_ious) # convert the iou's to an array
              max_iou_index = np.argmax(my_ious) # find the index of the max IOU (think of this as the "max iou index" of the "matching image indices" of the original list")
                     
              
                     
              max_iou = my_ious[max_iou_index] # the value of the max IOU
              #print('Detection #',det_num,' iou=',max_iou)
    
              if max_iou < iou_thresh: # if the max IOU doesn't pass the IOU threshold, it's background 
                #fp[st,class_index] = fp[st,class_index] + 1
                #pass
                confm[0,class_index,st] = confm[0,class_index,st] + 1
              
              else: # max_iou >= iou_thresh: # if the max IOU passess the IOU threshold, then add one to the appropriate target class, detection class, and score threshold
                #tp[st,class_index] = tp[st,class_index] + 1 # award TP
                #fn[st,class_index] = fn[st,class_index] - 1 # remove a FN
                target_class = gt_class[ind_im_id[max_iou_index]]
                target_class_index = np.where(np.array(class_list) == target_class)[0][0]
                confm[target_class_index+1,class_index,st] = confm[target_class_index+1,class_index,st] + 1
              
              
    
            else: # if the detection image_id and class match classes on this image, find the intersection over union for each potential TP
              #print('matching image & class found for det#',det_num,' searching for acceptable IOU')
              #print(ind_of_ind)
              #print(gt_im_id[ind_im_id[ind_of_ind]])
              #print(gt_class[ind_im_id[ind_of_ind]])
    
              my_ious = []
              # 3) find the IOU's for all the matching groundtruths
              for my_image_class_bboxes in gt_bbox[ind_im_id[ind_of_ind]]: # loop over each possible TP and find the IOU
                my_ious.append(iou(det_bbox[det_num],my_image_class_bboxes)) # append the IOU vlaue to this list
    
              my_ious = np.asarray(my_ious) # convert the iou's to an array
              max_iou_index = np.argmax(my_ious) # find the index of the max IOU (think of this as the "max iou index" of the "matching class indices" of the "matching image indices" of the original list")
              max_iou = my_ious[max_iou_index] # the value of the max IOU
              #print('Detection #',det_num,' iou=',max_iou)
    
              if max_iou < iou_thresh: # if the max IOU doesn't pass the IOU threshold, it's a false positive and background
                fp[st,class_index] = fp[st,class_index] + 1
                confm[0,class_index,st] = confm[0,class_index,st] + 1
              
              else: # max_iou >= iou_thresh: # if the max IOU passess the IOU threshold, it's a true positive (ie., this detection is on the same image, with the same class, with an acceptable iou) and the target should match the selection
                tp[st,class_index] = tp[st,class_index] + 1 # award TP
                fn[st,class_index] = fn[st,class_index] - 1 # remove a FN
                     
                target_class = gt_class[ind_im_id[ind_of_ind[max_iou_index]]]
                target_class_index = np.where(np.array(class_list) == target_class)[0][0]
                if target_class_index == class_index:
                     pass
                else:
                     print('target class index is ', target_class_index, 'detection class is ', class_index)
                #confm[target_class+1,class_index,st] = confm[target_class+1,class_index,st] + 1
                confm[target_class_index+1,class_index,st] = confm[target_class_index+1,class_index,st] + 1
    
                # Remove this GT index from the GT list so it can't be used again in the evaluation (at the current score threshold)
                orig_index = ind_im_id[ind_of_ind[max_iou_index]] # get back to the original index of the list
                gt_bbox = np.delete(gt_bbox,orig_index,axis=0)
                gt_im_id = np.delete(gt_im_id,orig_index,axis=0)
                gt_class = np.delete(gt_class,orig_index,axis=0)
                
        #pbar.update()
    #pbar.close()
    # do the precision, recall and F1 after all the TP, FP, and FN have been accounted for
    print('')
    print('done, computing p r and f1')
    # initialize the matrices as NANs incase there are results where the metrics can't be computed
    p=np.zeros((len(score_thresh),len(class_list)))*np.nan
    r=np.zeros((len(score_thresh),len(class_list)))*np.nan
    f1=np.zeros((len(score_thresh),len(class_list)))*np.nan
    for st in np.arange(0,len(score_thresh)): # for each score threshold row
      for class_num in np.arange(0,len(class_list)): # for each class column
        
        if (tp[st,class_num]+fp[st,class_num]) == 0: # check if the precision demominator will be zero
          pass
        else:
          p[st,class_num] = tp[st,class_num]/(tp[st,class_num]+fp[st,class_num]) # precision
    
        if (tp[st,class_num]+fn[st,class_num]) == 0: # check if the recall denominator will be zero
          pass
        else:
          r[st,class_num] = tp[st,class_num]/(tp[st,class_num]+fn[st,class_num]) # recall
    
        if np.isnan(p[st,class_num]+r[st,class_num]): # if either P or R are nan
          pass
        elif (p[st,class_num]+r[st,class_num]) == 0: # if the F1 denominator is zero
          pass
        else:
          f1[st,class_num] = 2*(p[st,class_num]*r[st,class_num])/(p[st,class_num]+r[st,class_num]) # f1
    
    # find the average metrics for each score threshold
    #p_mean = np.nanmean(p,axis=1)
    #r_mean = np.nanmean(r,axis=1)
    #f1_mean = (2*p_mean*r_mean)/(p_mean+r_mean)
    #f1_mean = np.nanmean(f1,axis=1)
    #tp_sum = np.sum(tp,axis=1)
    #fp_sum = np.sum(fp,axis=1)
    #fn_sum = np.sum(fn,axis=1)
    print('done')
    
    return p,r,f1,tp,fp,fn,confm