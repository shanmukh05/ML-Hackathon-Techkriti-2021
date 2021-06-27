'''
---Run the following python file to predict the results of seg_pred data
---Change the path of TEST_PATH and MODEL_PATH accordingly.
--- Make sure that all the dataset folders (seg_pred,seg_train,seg_test are in dataset folder.
'''

#Importing necessary Libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf

#Initializing necessary constants
MAIN_PATH = os.getcwd()
MAIN_PATH = MAIN_PATH[:-5]      #./codes[:-5] == ./

TEST_PATH = os.path.join(MAIN_PATH,"dataset/seg_test/seg_test")
HEIGHT,WIDTH = 150,150
MODEL_PATH = os.path.join(MAIN_PATH,"model/best_model.h5")
NUM_IMAGES = 0

#counting number of images
for folder in os.listdir(TEST_PATH):
  NUM_IMAGES+= len(os.listdir(os.path.join(TEST_PATH,folder)))

#Preparing pred data for prediction
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
   )
test_ds = test_datagen.flow_from_directory(
                      TEST_PATH,
                      target_size = (HEIGHT,WIDTH),
                      shuffle = False,
                      batch_size = NUM_IMAGES
                       )

classes_dict = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
model = tf.keras.models.load_model(MODEL_PATH)

# predicting on test data
for batch in test_ds:
    img_arr,labels = batch
    labels = np.array(labels)
    val_pred = model.predict(img_arr)
    break

# Accuracy calculation   
val_predictions = [np.argmax(i) for i in val_pred]
val_labels = [np.argmax(i) for i in labels]
bool_arr = []
binary_ls  = []

for i in range(NUM_IMAGES):
    if val_predictions[i] == val_labels[i]:
        bool_arr.append(1)
        binary_ls.append("correct")
    else:
        bool_arr.append(0)
        binary_ls.append("incorrect")

test_accuracy = sum(bool_arr)/len(bool_arr)
print("Test Accuracy: ",test_accuracy)

#preparing csv file
predictions = [classes_dict[i] for i in val_predictions]

pred_df = pd.DataFrame(val_labels, columns=['original_label'])
pred_df["predicted_value"]  = val_predictions
pred_df["result"]    = binary_ls
pred_df.to_csv('./test.csv',index=False)