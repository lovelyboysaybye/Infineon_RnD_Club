"""ATSE"""

import numpy as np			    #---First group of imports----
import tensorflow as tf 		#---is third party libraries--
                        		#---that installed by pip-----

import json   					#---Second group of imports---
import pathlib   				#---is standart libraries.----
import os						#-----------------------------
import sys                      #----------------------------

folder = '.\\convolutional_model'
path = r"D:\WORK_TF\Model_Speech\Model_Speech\convolutional_model\\"

model = tf.keras.models.load_model(path) #---Load saved model

converter = tf.lite.TFLiteConverter.from_keras_model(model) 

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]  
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8


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
    
    i = 0
    datas = []
    labels = []
    for file in files_names:
        with open (path_data + file) as training_files:
            json_train = json.load(training_files)
    
        data = np.array(json_train['values']).astype(np.float32)
        data = data[:16000]
        data = np.array_split(data, json_train['frame_num'])
        data = np.array(data)
        data1 = []

        if (json_train['label'] == 1):
            label = np.array([1, 0])
        else:
            label = np.array([0, 1])
        
        datas.append(data)
        labels.append(label)
        i = i + 1
        print(i)

    datas = np.array(datas).astype(np.float32)
    labels = np.array(labels).astype(np.float32)

    return datas, labels


""" Generate dataset for spectrogram  """
""" values 0..65535 (not for 0..1)    """
""" like, when we traing our model.   """

def representative_dataset_gen():
    datas, labels = get_data(r"D:\WORK_TF\dataset\ATSE_Corrected\JSON\all_all\\")
    training_datas =  np.expand_dims(datas, axis = 3)
    my_ds = tf.data.Dataset.from_tensor_slices((training_datas)).batch(1)

    for input_value  in my_ds.take(len(datas)):
        yield [input_value]

converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
tflite_models_dir = pathlib.Path("Tflite_Model")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"converted_model.tflite"
tflite_model_file.write_bytes(tflite_model)

#After it - check size of generated .tflite file of your model.

