# Dataset
 - Coco
    Wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    Wget http://images.cocodataset.org/zips/train2014.zip 
    Wget http://images.cocodataset.org/zips/val2014.zip
    Wget http://images.cocodataset.org/zips/test2014.zip
    Wget http://images.cocodataset.org/zips/test2015.zip
    Wget http://images.cocodataset.org/zips/train2017.zip 
    Wget http://images.cocodataset.org/zips/val2017.zip 
    Wget http://images.cocodataset.org/zips/test2017.zip
    Wget http://images.cocodataset.org/zips/unlabeled2017.zip 

    Annotations
    Wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    Wget http://images.cocodataset.org/annotations/image_info_test2014.zip 
    Wget http://images.cocodataset.org/annotations/image_info_test2015.zip
    Wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 
    Wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip 
    Wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
    Wget http://images.cocodataset.org/annotations/image_info_test2017.zip
    Wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip


- Visual Genome
   https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
   - v1.2 image Part1: Wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
   - v1.2 image Part2: Wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

   Caption
   - Wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip
      - vg.json

# ALBEF
 - https://github.com/salesforce/ALBEF?tab=readme-ov-file

# Traing
 - start a new training
 ```
 python vlm_train.py
 ```

 - continue training from a given checkpoint
 ```
 python vlm_train.py --checkpoint /Users/chengbai/ml/cheng_git/notebooks/vlm_caption_model_0_20240927_073655_2500.pt
 ```