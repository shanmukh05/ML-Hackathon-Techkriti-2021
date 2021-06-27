'''
---Run the following python file to predict the results of seg_pred data
---Change the path of PRED_PATH and MODEL_PATH accordingly.
--- Make sure that all the dataset folders (seg_pred,seg_train,seg_test are in dataset folder.
'''

#Importing necessary Libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf

#Initializing necessary constants
MAIN_PATH = os.getcwd()
MAIN_PATH = MAIN_PATH[:-5]       #./codes[:-5] == ./

PRED_PATH = os.path.join(MAIN_PATH,"data/seg_pred")
HEIGHT,WIDTH = 150,150
MODEL_PATH = os.path.join(MAIN_PATH,"model/best_model.h5")
NUM_IMAGES = len(os.listdir(os.path.join(PRED_PATH,"seg_pred")))


#Preparing pred data for prediction
pred_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
   )
pred_ds = pred_datagen.flow_from_directory(
                      PRED_PATH,
                      target_size = (HEIGHT,WIDTH),
                      shuffle = False,
                      batch_size = 32
                       )

classes_dict = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

#loading the model and predicting
model = tf.keras.models.load_model(MODEL_PATH)
pred = model.predict(pred_ds)
predictions = [classes_dict[np.argmax(i)] for i in pred]

#preparing csv file
pred_df = pd.DataFrame(predictions, columns=['predictions'])
pred_df.to_csv('./prediction.csv',index=False)