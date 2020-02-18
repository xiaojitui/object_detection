# YOLO

This is the YOLO_v3 model to detect table areas. 

This is developed based on: https://github.com/YunYang1994/tensorflow-yolov3
Please go to the link, read for more information, and git clone. 

<br><br>
To train and test your own dataset, follow the instruction: 
- (1) save class names as 'dataset.names' in './data/classes'
- (2) save train_data groundtruth and test_data groundtruth as 'train.txt' and 'test.txt' in './data/dataset'. 
    - The format is: imgpath, bbox, classid. In this case, classid = 1 (table) or 0 (background)
- (3) To train, run: 
    - python convert_weight.py --train_from_coco
    - python train.py
- (4) To evaluate, run:
  - python evaluate.py
  - cd mAP
  - python main.py -na

