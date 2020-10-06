# Micro Speech Neural Network repository

# Requirements:
  * tensorflow             2.3.1

# TensorFlow model - Float_NN.py
  ## *Tensorflow 2.X - Convolutional model for recognise Tak/Ni spectrogram.*
  By using multiprocessing, now we can prepare our dataset **x4 - x8** faster!
  
  
# Converting Saved Trained TensorFlow model to .tflite formal - to_tflite.py
  You can convert your saved model using this file.
  After converting, your model will have int8 type weights.
  *Note, that input and output type still have float type.*
  We use our dataset again, to change back dataset range for 0..65535.  
  
# Evaluating converted .tflite model - tflite_evaluate.py
  Evaluate accuracy of your .tflite model. Please, compare .tflite accuracy of your model to previous not converted model.
  
# Our dataset - TAK_NI_Dataset_ATSE_YABE.zip
