import librosa
import numpy as np
from tensorflow._api.v2.test import is_built_with_cuda, gpu_device_name
from tensorflow.python.keras import models
import soundfile as sf
from tqdm import tqdm
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

is_cuda = is_built_with_cuda()
is_gpu = gpu_device_name()
model_path = os.path.join(os.getcwd(),'saved','models', 'MSE_FCNN_AN.model')
error = None
if not os.path.exists(model_path):
    print("Error. Model or Weights not found. Please download the model and weights from the repository and place them in the saved/models folder. Alternatively, you can run the train the model on your own data.")
    exit()
model = models.load_model(model_path)
print("Model Available: " + ('\u274C', '\u2705')[int(model is not None)])
print("Cuda Available: " + ('\u274C', '\u2705')[int(is_cuda)])
print("GPU Available: " + ('\u274C', '\u2705')[int(is_cuda)])

def DeepDenoise(filename, sr=16000, segment_time=0):
    outfile = filename.split('.')[0] + '_ml_denoised.wav'
    # Segment time is used to split the audio into segments of X seconds for inference
    # SR = 16000 is what i used for training as specified in the paper
    # So, I'm assuming that's what is appropriate for the inference as well
    x, _ = librosa.load(filename, sr=sr)
    # print(x.shape)
    # Implement split into segments, its okay if the last segment is less than 1.5 seconds.
    if segment_time > 0:
        max_samples = int(sr*segment_time)
        x_out = []
        for i in (range(0, len(x), max_samples)):
            x_i = x[i:i+max_samples]
            # print (x_i.shape)
            x_i = np.expand_dims(np.expand_dims(x_i, axis=0), -1)
            x_i_out = model.predict(x_i)
            # print (x_i_out.shape)
            x_out.append(x_i_out)
        x_out = np.concatenate(x_out, axis=1).squeeze()
        sf.write(outfile, x_out, sr)
    if segment_time == 0:
        x_out = model.predict(np.expand_dims(np.expand_dims(x, axis=0), -1)).squeeze()
        sf.write(outfile, x_out, sr)
    # print(x_out.shape)
