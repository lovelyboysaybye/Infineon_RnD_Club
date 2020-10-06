"""ATSE"""
import numpy as np			    #---First group of imports----
import tensorflow as tf 		#---is third party libraries--
                         		#---that installed by pip-----

import json   					#---Second group of imports---
import datetime 				#---is standart libraries.----
import os						#-----------------------------
import subprocess
from multiprocessing import cpu_count, Pool, current_process

#import subprocess 				#---Import for .bat script,---
								#---that clears logging_folder
								#---for TensorBoard.----------
								#---NOT USED IN THIS PROJECT--

""" Special .bat script for TensorBoard word."""
#subprocess.Popen([r'bat\all.bat'])
#logdir = r'C:\LoggingDirrectory'
#callbacks = tf.keras.callbacks.TensorBoard(log_dir=logdir, embeddings_freq = 1,  		\
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

Path_To_Dataset = r"F:\\WORK_TF\\dataset_ATSE_YABE\\"	

def Read_Samples_Dataset(file):
    with open(file) as dataset_sample:
        json_train = json.load(dataset_sample)

    data = np.array(json_train['values']).astype(np.float)
    data = data[:16000]             # ---if use bad dataset with 16003 in name
    data = np.array_split(data, json_train['frame_num'])
    data = np.array(data)

    if (json_train['label'] == 1):  # ---Make one-hot vector
        label = np.array([1, 0])    # ---for categorical_crossentropy.
    else:                           # ---For future widest labels dataset
        label = np.array([0, 1])    # ---we use categorical_crossentropy

    return data, label

""" Split on Training and Validation  """
"""-----------------------------------"""
""" JSON file of data contains        """
""" 			"trainig_size":       """
"""                         0..1      """

def Split_Datasets(dataset, training_size = 0.9):
    datas, labels = [], []
    for couple in dataset:
        datas.append(couple[0])
        labels.append(couple[1])

    datas = np.array(datas).astype(np.float)                    #---For better training rescale
    labels = np.array(labels).astype(np.float)                  #---datas to 0..1 float-range

    training_datas = datas[:int(training_size * len(datas))]
    training_labels = labels[:int(training_size * len(labels))]

    testing_datas = datas[len(training_datas):]
    testing_labels = labels[len(training_labels):]

    training_datas = np.expand_dims(training_datas, axis = 3)	#---Our data is 3-dimension array,
    testing_datas = np.expand_dims(testing_datas, axis = 3)		#---Conv2D expecting 4-dimension array
    
    training_datas /= 65535 	                                #---For better training rescale
    testing_datas /= 65535 		                                #---datas to 0..1 float-range

    return [[training_datas, training_labels], [testing_datas, testing_labels]]


if __name__ == '__main__':
    """ ---Get list of samples name---------- """
    input_array = [Path_To_Dataset + file for file in os.listdir(Path_To_Dataset)]      
                                            
    """ ---Parallel processing of input data- """
    """ ---with available cpu`s-------------- """
    with Pool(processes=cpu_count()) as p:
        dataset = list(p.map(Read_Samples_Dataset, input_array))

    training_dataset, testing_dataset = Split_Datasets(dataset)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(1, (3, 3), input_shape=(250, 64, 1)),                                               
        tf.keras.layers.MaxPooling2D(2, 2),                                                                 
        tf.keras.layers.ReLU(),                                               
    
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='softmax')
        ])


    model.summary()		#---Print model architecture to console
    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(x = training_dataset[0], y = training_dataset[1], epochs=15, validation_data = tuple(testing_dataset), batch_size = 32, verbose = 1 \
	    #, callbacks=[callbacks]
	    #Callbacks used for TensorBoard
	    )

    folder = 'convolutional_model'
    if (os.path.exists(folder) == False):
        os.makedirs(name=folder)

    count = 0
    for path in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, path)):
            count += 1
    current_model = os.path.join(folder, str(count))

    os.makedirs(name=current_model)
    model.save(current_model) #path to folder