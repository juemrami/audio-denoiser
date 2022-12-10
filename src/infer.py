import librosa
import numpy as np
from tensorflow._api.v2.test import is_built_with_cuda, gpu_device_name
from tensorflow.python.keras import models
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

is_cuda = is_built_with_cuda()
is_gpu = gpu_device_name()
model_path = os.path.join(os.getcwd(),'saved','models', 'fcnn_AN.model')
error = None

print("Cuda Available: " + ('\u274C', '\u2705')[int(is_cuda)])
print("GPU Available: " + ('\u274C', '\u2705')[int(is_cuda)])
if not os.path.exists(model_path):
    print("Error. Model or Weights not found. Please download the model and weights from the repository and place them in the saved/models folder. Alternatively, you can run the train the model on your own data.")
model = models.load_model(model_path)

def DeepDenoise(filename, sr=16000):
    # SR = 16000 is what i used for training as specified in the paper
    # So, I'm assuming that's what is appropriate for the inference as well
    x, _ = librosa.load(filename, sr=sr)
    prediction = 
    pass
