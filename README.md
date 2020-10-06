# Micro Speech Neural Network repository

## Requirements:
  *Pip requirements:*
  * tensorflow             2.3.1
  
  If using Visual Studio, be carefull for Starting .py file without Debug mode.
  It necessary for using multiprocessing (you're not capable to Debug multithreading and multiprocessing).



## Float_NN.py - TensorFlow model
  #### *Tensorflow 2.X - Convolutional model for recognise Tak/Ni spectrogram.*
  By using multiprocessing, now we can prepare our dataset **x4 - x8** faster!
  Prepare training couples: *datas* - array of 16000 float in range 0..65535
                            *labels* - [1, 0] equal to "Yes"
                                     - [0, 1] equal to "No"
  For training, we rescalling array *datas* to 0..1 range for better perfmonace.
  
  
## Quantized.py - Quantize and convert to .tflite
  Quantized our model to int8 type of weights and output.
  **!!!PROBLEM!!!**
  If set input_type to tf.int8, when evaluating model we will have
  random output of our model.

  Range of input array rescalling to 0..127 
  
## Evaluate_Quantized.py
  Range of input array rescalling to 0..127
  For evaluating model, we taking abs of subtracting Labels and Predictions, both of them divided by two.
  If result is closer to 0 - prediction is correct, if closer to 127 - prediction is uncorrect.
  
## Our dataset - TAK_NI_Dataset_ATSE_YABE.zip
