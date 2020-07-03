"""ATSE"""

import numpy as np			    #---First group of imports----
import tensorflow as tf 		#---is third party libraries--
                        		#---that installed by pip-----

import json   					#---Second group of imports---
import os          				#---is standart libraries.----



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


interpreter_quant = tf.lite.Interpreter(model_path = str(r'D:\WORK_TF\Model_Speech\Model_Speech\Tflite_Model\converted_model.tflite'))
interpreter_quant.allocate_tensors()
input_index = interpreter_quant.get_input_details()[0]["index"]
output_index = interpreter_quant.get_output_details()[0]["index"]

datas, labels = get_data(r'D:\WORK_TF\dataset\ATSE_Corrected\JSON\all_all\\')
datas = np.expand_dims(datas, axis = 3)

accuracy = 0
predictions = []
how_many = len(datas)
diff = []

for index in range(how_many):
    value = np.reshape(datas[index], (1, 250, 64, 1))
    interpreter_quant.set_tensor(input_index, value)
    interpreter_quant.invoke()
    predictions.append(interpreter_quant.get_tensor(output_index))
    print(labels[index])
    print(predictions[-1])

    diff.append(abs(predictions[-1][0] - labels[index]))
    #print(diff[-1])
    #print('\n\n')


for index in range(how_many):
    if (diff[index][0] < 0.5 and diff[index][1] < 0.5):
        accuracy += 1
        print("Iter {}: get".format(index))
        continue
    print("\n\noops\n\n")

print("Accuracy: {}".format(accuracy / how_many))

#Accuracy after converting your model to .tflite with int8 weights