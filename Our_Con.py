"""ATSE"""

import numpy as np			    #---First group of imports----
import tensorflow as tf 		#---is third party libraries--
from pprint import pprint 		#---that installed by pip-----

import json   					#---Second group of imports---
import datetime 				#---is standart libraries.----
import os						#-----------------------------

#import subprocess 				#---Import for .bat script,---
								#---that clears logging_folder
								#---for TensorBoard.----------
								#---NOT USED IN THIS PROJECT--

""" Special .bat script for TensorBoard word."""
# subprocess.Popen([r'bat\all.bat'])
# logdir = r'C:\LoggingDirrectory'
# callbacks = tf.keras.callbacks.TensorBoard(log_dir=logdir, embeddings_freq = 1,  		\
#											 profile_batch = 0, histogram_freq = 1, 	\
#											 write_graph = True, write_images= True,	\
# 											 update_freq = 'epoch')


""" Loading and preprocessing dataset """
"""-----------------------------------"""
""" JSON file of data contains        """
""" 			"label":	0 - No;   """
""" 						1 - Yes;  """
""" 			"values": array of    """
"""						  16000	      """
"""						  spectrogram """
""" 					  values. 	  """
"""									  """
""" 			"frame_num": number of"""
"""						     frames   """
"""									  """
"""				"frame_size": size of """
"""							  frame   """
							
def get_data(path_data):
    files_names = os.listdir(path_data)
    
    print("\n\nStart preprocessing datas for {0} dataset examples.\n".format(len(files_names)))

    i = 0
    datas = []
    labels = []

    for file in files_names:
        with open (path_data + file) as training_files:
            json_train = json.load(training_files)
    
        data = np.array(json_train['values']).astype(np.float)

        data = data[:16000] #---if use bad dataset with 16003 in name

        data = np.array_split(data, json_train['frame_num'])
        data = np.array(data)
        data1 = []

        if (json_train['label'] == 1):	#---Make one-hot vector
            label = np.array([1, 0]) 	#---for categorical_crossentropy.
        else:							#---For future widest labels dataset
            label = np.array([0, 1])	#---we use categorical_crossentropy
        
        datas.append(data)
        labels.append(label)
        i = i + 1
        print(i)

    datas = np.array(datas).astype(np.float)
    labels = np.array(labels).astype(np.float)

    return datas, labels


datas, labels = get_data(r"D:\WORK_TF\dataset\ATSE_Corrected\JSON\all_all\\")
print(datas.shape)
print(labels.shape)

training_datas = datas[:int(0.9 * len(datas))]
training_labels = labels[:int(0.9 * len(labels))]

testing_datas = datas[len(training_datas):]
testing_labels = labels[len(training_labels):]


training_datas =  np.expand_dims(training_datas, axis = 3)	#---Our data is 3-dimension array,
testing_datas = np.expand_dims(testing_datas, axis = 3)		#---Conv2D expecting 4-dimension array

training_datas = training_datas / 65535						#---For better training rescale
testing_datas = testing_datas / 65535 						#---datas to 0..1 float-range

print(training_datas.shape)
print(training_labels.shape)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(1, (3, 3), input_shape=(250, 64, 1)),                                               
    tf.keras.layers.MaxPooling2D(2, 2),                                                                 
    tf.keras.layers.ReLU(),                                               
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax')
    ])


model.summary()		#---Print model architecture to console

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x = training_datas, y = training_labels, epochs=15, validation_data = (testing_datas, testing_labels), batch_size = 32, verbose = 1 \
	#, callbacks=[callbacks]
	# Callbacks used for TensorBoard
	)

model.save('convolutional_model') #path to folder