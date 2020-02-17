This is the Faster-RCNN model to detect table areas. 
*note: the model is developed on Keras-2.2.4. You can download this version by running: pip install --user keras==2.2.4


To train the model, run:
python frcnn_train.py

After training, the results will be saved in the 'saved_record' folder. The folder includes a 'model_frcnn.hdf5' file storing weights of the trained model, and a 'config.pickle' file summarizing training parameters.


To do prediction, put images into the '../test_data' folder, then run:
python frcnn_predict.py

The prediction results will be saved in the '../test_result' folder. The folder includes the detected table boundaries drawn on the original images, and a TXT file called 'preresuls.txt' summarizing the coordinates of the table boundaries.
 
