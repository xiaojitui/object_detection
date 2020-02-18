# object_detection

These are two Object Detection models to detect table areas in an image or a PDF*. 

*note: if the input is PDF format, you need to use 'pdf_to_img.py' to convert your PDF to image first. 

The two models are: Faster-RCNN (in 'frcnn' folder) and YOLO_v3 (in 'yolo' folder) 
- The FRCNN model is developed based on two references: https://github.com/jinfagang/keras_frcnn and https://github.com/kbardool/keras-frcnn
- The YOLO model is developed based on the reference: https://github.com/YunYang1994/tensorflow-yolov3
The FRCNN model is slower than the YOLO model, but can provide higher accuracy. 


The outputs of the models include:
- detected table boundaries drawn on the original images
- table boundary boxes (in a txt file), in the format of [x1, y1, x2, y2]*. 

*note: origin = [0, 0, 0, 0] is at the top-left corner of the image. 


The datasets used to train the models include:
- ICDAR 2013 Table Completion Dataset (http://www.tamirhassan.com/html/competition.html). 
 -- The dataset is the format of pdf and with ground truth of table boundaries. The dataset is converted from pdf format to image format with 'pdf_to_img.py'
- PaleoDocs. 
 -- The dataset can be downloaded at: https://github.com/HazyResearch/pdftotree (in section: 'Example Dataset: Paleontological Papers'). The dataset is converted from pdf format to image format with 'pdf_to_img.py'
- UNLV Dataset. 
 -- The dataset can be downloaded at: http://www.iapr-tc11.org/mediawiki/index.php?title=Table_Ground_Truth_for_the_UW3_and_UNLV_datasets


To train the models, training data should be put in the 'train_data' folder and the ground truth should be saved in the 'ground_truth.txt' with the format shown in the file. The test data should be put in the 'test_data' folder. 
