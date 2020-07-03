# Infineon_RnD_Club
ATSE Infineon repository

# TensorFlow model - Our_Con.py
  Tensorflow 2.X - Convolutional model for recognise Tak/Ni spectrogram.
  You should try other architecture of model, and write comparasion of them.
  For faster and better training, we scaling our dataset values to 0..1 float range.
  
# Converting Saved Trained TensorFlow model to .tflite formal - to_tflite.py
  You can convert your saved model using this file.
  After converting, your model will have int8 type weights.
  *Note, that input and output type still have float type.*
  We use our dataset again, to change back dataset range for 0..65535.  
  
# Evaluating converted .tflite model - tflite_evaluate.py
  Evaluate accuracy of your .tflite model. Please, compare .tflite accuracy of your model to previous not converted model.
