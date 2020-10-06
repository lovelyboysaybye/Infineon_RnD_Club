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
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  conv2d (Conv2D)              (None, 248, 62, 1)        10
  _________________________________________________________________
  max_pooling2d (MaxPooling2D) (None, 124, 31, 1)        0
  _________________________________________________________________
  re_lu (ReLU)                 (None, 124, 31, 1)        0
  _________________________________________________________________
  flatten (Flatten)            (None, 3844)              0
  _________________________________________________________________
  dense (Dense)                (None, 2)                 7690
  =================================================================
  Total params: 7,700
  Trainable params: 7,700
  Non-trainable params: 0
  
  
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
