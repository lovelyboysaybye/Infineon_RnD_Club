"""ATSE"""

import numpy as np			    #---First group of imports----
import tensorflow as tf 		#---is third party libraries--
                        		#---that installed by pip-----

import json   					#---Second group of imports---
import pathlib   				#---is standart libraries.----
import os						#-----------------------------
from multiprocessing import cpu_count, Pool, current_process

folder = '.\\convolutional_model'
path = f".\convolutional_model\{len(os.listdir(folder)) - 1}\\"
#path = f".\convolutional_model\0\\"

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

def Split_Datasets(dataset):
    datas, labels = [], []
    for couple in dataset:
        datas.append(couple[0])
        labels.append(couple[1])

    datas = np.expand_dims(datas, axis = 3)	#---Our data is 3-dimension array,
    datas = datas.astype(np.float)

    return datas, labels


""" Generate dataset for spectrogram  """
""" values 0..127 (not for 0..1)    """
""" like, when we traing our model.   """

def representative_dataset_gen():
    """ ---Get list of samples name---------- """
    input_array = [Path_To_Dataset + file for file in os.listdir(Path_To_Dataset)]      
                                            
    """ ---Parallel processing of input data- """
    """ ---with available cpu`s-------------- """
    with Pool(processes=cpu_count()) as p:
        dataset = list(p.map(Read_Samples_Dataset, input_array))
    
    datas, labels = Split_Datasets(dataset)
    """ -------------ATTENTION--------------- """
    """ ---Only there we can change type----- """
    """ ---and range of our values!---------- """
    datas = np.array(datas).astype(np.float32) / 512 # ATTENTION! Onl
    
    my_ds = tf.data.Dataset.from_tensor_slices(datas).batch(1)

    for input_value  in my_ds.take(len(datas)):
        yield [input_value]

if __name__ == '__main__':
    model = tf.keras.models.load_model(path) #---Load saved model

    converter = tf.lite.TFLiteConverter.from_keras_model(model) 
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]  
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    """ ---Set data type of input and output- """
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.int8
   
    converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()
    subfold = path + "Converted"
    if (os.path.exists(subfold) == False):
        os.makedirs(name=subfold)

    tflite_models_dir = pathlib.Path(subfold)
    tflite_model_file = tflite_models_dir/"converted_model.tflite"

    tflite_model_file.write_bytes(tflite_model)
    #After it - check size of generated .tflite file of your model.

