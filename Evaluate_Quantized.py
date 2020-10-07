"""ATSE"""

import numpy as np			    #---First group of imports----
import tensorflow as tf 		#---is third party libraries--
                        		#---that installed by pip-----

import json   					#---Second group of imports---
import os          				#---is standart libraries.----
from multiprocessing import cpu_count, Pool, current_process

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
        label = np.array([127, -128])    # ---for categorical_crossentropy.
    else:                           # ---For future widest labels dataset
        label = np.array([-128, 127])    # ---we use categorical_crossentropy

    return data, label

def Split_Datasets(dataset):
    datas, labels = [], []
    for couple in dataset:
        datas.append(couple[0])
        labels.append(couple[1])

    datas = np.expand_dims(datas, axis = 3)	#---Our data is 3-dimension array,
    datas = datas.astype(np.float)

    return datas, labels


if __name__ == "__main__":
    interpreter_quant = tf.lite.Interpreter(model_path = str(r'F:\WORK_TF\PythonApplication1\PythonApplication1\convolutional_model\12\Converted\converted_model.tflite'))
    interpreter_quant.allocate_tensors()

    input_index = interpreter_quant.get_input_details()[0]["index"]
    output_index = interpreter_quant.get_output_details()[0]["index"]

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
    datas = (np.array(datas).astype(np.float32) / 256 - 128).astype(np.int8)   # last .astype() - convert to correct type
    labels= np.array(labels).astype(np.int8)

    accuracy = 0
    predictions = []
    how_many = len(datas)
    diff = []

    for index in range(how_many):
        value = np.reshape(datas[index], (1, 250, 64, 1))
        interpreter_quant.set_tensor(input_index, value)
        interpreter_quant.invoke()
        predictions.append(interpreter_quant.get_tensor(output_index))

        """ ---For Evaluating model int8 range """
        """ ---(-128..127) we reduce by 2 times"""
        """ -----------------------------------"""
        """ ---If Prediction and Label has same"""
        """ ---values, difference will be close"""
        """ ---to zero value.------------------"""
        """ -----------------------------------"""
        """ ---On the other hand, differnce of-"""
        """ ---Label and Prediction will going-"""
        """ ---to max int8 value: 127----------"""
        """ -----------------------------------"""
        """ - Examples:                        """
        """ -         Label = [127, -128]      """
        """ -         Prediction = [-128, 127] """
        """ -                                  """
        """ - Label // 2 - Prediction //2 =    """
        """ - =[63, -64]-[-64, 63] = [127, 127]"""
        """ -           Result: Bad Prediction """
        diff.append(abs(np.subtract(labels[index] // 2, predictions[-1][0] // 2)))


    for index in range(how_many):
        if (diff[index][0] < 64 and diff[index][1] < 64):
            accuracy += 1
        print(f"L: {labels[index] // 2}; P: {predictions[index][0] // 2}; Diff: {diff[index]}")

    print("Accuracy: {}".format(accuracy / how_many))

    #Accuracy after converting your model to .tflite with int8 weights