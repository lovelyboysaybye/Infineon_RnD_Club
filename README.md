# Micro Speech Neural Network repository

## Requirements:
  *Pip requirements:*
  * tensorflow             2.3.1
  
  If using Visual Studio, be carefull for Starting .py file without Debug mode.
  It necessary for using multiprocessing (you're not capable to Debug multithreading and multiprocessing).

## TensorFlow model - Float_NN.py
  #### *Tensorflow 2.X - Convolutional model for recognise Tak/Ni spectrogram.*
  By using multiprocessing, now we can prepare our dataset **x4 - x8** faster!
  Prepare training couples: *datas* - array of 16000 float in range 0..65535
                            *labels* - [1, 0] equal to "Yes"
                                     - [0, 1] equal to "No"
  For training, we rescalling array *datas* to 0..1 range for better perfmonace.
  
  
  
## Converting Saved Trained TensorFlow model to .tflite formal - to_tflite.py
  You can convert your saved model using this file.
  After converting, your model will have int8 type weights.
  *Note, that input and output type still have float type.*
  We use our dataset again, to change back dataset range for 0..65535.  
  
## Evaluating converted .tflite model - tflite_evaluate.py
  Evaluate accuracy of your .tflite model. Please, compare .tflite accuracy of your model to previous not converted model.
  
## Our dataset - TAK_NI_Dataset_ATSE_YABE.zip
