<!-- README.md -->
# CleanMachine - Speech Denoising App
<!-- name: "CleanMachine" -->
## Description
<!-- PROJECT DESCRIPTION -->
CleanMachine allows you to pass it a `.wav` file containing noisy speech and denoise it using one of 3 algorithms:
1. Spectral Subtraction.
2. Wiener Filter
3. Machine Learning

The first two "traditional" algorithms are implemented based on the book: [*"Speech Enhancement: Theory and Practice"*](https://doi.org/10.1201/b14529) by Philipos C. Loizou.

The deep machine learning algorithm is implemented based on the paper [*"Speech Denoising with Deep Feature Losses"*](https://doi.org/10.48550/arXiv.1806.10522) by Francois G. Germain et al. (see [here](#deep-learning-model) to read more)

<!-- Mention the machine learning is implemented based on the paper  "Germain, Francois G., et al. Speech Denoising with Deep Feature Losses. arXiv:1806.10522, arXiv, 14 Sept. 2018. arXiv.org, https://doi.org/10.48550/arXiv.1806.10522." -->
<!-- And the other two algorithms come from techniques curated in "Loizou, Philipos C. Speech Enhancement: Theory and Practice. 2nd ed., CRC Press, 2013. DOI.org (Crossref), https://doi.org/10.1201/b14529." -->

## Installation

CleanMachine requires Python 3.9 and the following packages:
- tensorflow 2.10
- librosa 0.9.23
- soundfile (latest version)
- numpy (latest version)

## Usage
<!-- RUN INSTRUCTIONS -->
<!-- Pre-reqs: python 3.9, tensorflow, librosa, soundfile, numpy  -->
<!-- mention steps on how to run the denoise -->
<!-- the  CLI entry point is src/cm.py  -->
<!-- Takes 2 arguments at the moment  -->
<!-- 1. '-f' a relative or absolute pathname to the input wave file -->
<!-- Mention that the input is a wave file -->
<!-- 2. '-a' algorithm type (SS, WF, or ML) used for the de-noising task. Spectral Subtraction, Wiener Filter, or Machine Learning respectively' -->
<!-- the next one is still on the todo -->
<!-- 3. '-o' output file name (optional) -->
<!-- Mention that ouput is currently saved with the same name as the input file with `XX_denoised` appended to it where XX refers the lower case version of the -a arg -->
<!-- Mention that the output is saved in the same directory as the input file -->
<!-- Mention that the output is a wave file -->
To use CleanMachine, run the `cm.py` CLI entry point. Current available arguments are:
- `-f` : The "filename" argument should be a relative or absolute pathname to the input wave file that you want to denoise
- `-a`: The "algorithm" argument should specify the keyword algorithm that you want to use for the de-noising task.
    Currently, the supported algorithms are:
    - `"SS"` for spectral subtraction
    - `"WF"` for the Wiener filter
    - `"ML"` for machine learning
### todo:
- `-o` argument to specify the output file name.
### Example
To denoise a wave file called `input.wav` using the Wiener filter algorithm you would run the following command:
```bash
python cm.py -f input.wav -a WF
```
The output will be saved with the same name as the input file, with `"XX_denoised"` appended to it, where `XX` is the `-a` argument. In this case, the output file will be called `input_wf_denoised.wav`.




<!-- FILE CONTENTS -->
<!-- "./src/cm.py" as mentioned this is the just entry point for the CLI. -->
<!-- "./src/traditional.py" This file contains 2 functions that use traditional methods for speech denoisng. Namely Wiener Filter and Spectral Subtraction -->
<!-- insert Quick non technical explaination for WF and SS here -->
<!-- "./src/infer.py" This file contains 1 function that uses a machine learning model to denoise speech. -->
<!-- './models/DenoiseFeatureLossModels.ipynb' mention to check the "Deep Learning Model" section. -->
<!-- Mention that the individual modules can also be imported into other python project using `from traditional import WienerFilter, SpectralSubtraction` or `from infer import DeepDenoise` -->
## File Contents
The CleanMachine project contains the following files:
- [src/cm.py](./src/cm.py) : This is the entry point for the CleanMachine command-line interface (CLI).
- [src/traditional.py](./src/traditional.py): This file contains the two functions that use traditional methods for speech denoising, namely Wiener Filtering and Spectral Subtraction.
- [src/infer.py](./src/infer.py) : This file contains the function that uses tensorflow to load the trained machine learning model to denoise speech.
- [src/models/DenoiseFeatureLossModels.ipynb](./src/models/DenoisFeatureLossModels.ipynb) : This Jupyter notebook contains the code for the machine learning model used by the infer.py module. This is the notebook that was used to train the model and save it to the ./models directory. (See the "Deep Learning Model" section below for more details.)
- [src/saved/models\](./src/saved/models): This directory contains some of the saved machine learning models that can be used by the `infer.py` module. Different models using different loss functions and parameters are saved for comparison purposes during development(so currently) of the final model. The models are saved in the [SavedModel](https://www.tensorflow.org/guide/saved_model) format.
- [src/scripts/**](./src/scripts/) and [deprecated/**](./deprecated/) : These directories contains various scripts for testing code used throughout the course of this project. Such as the scripts used to generate the the training data for the machine learning models. **TO BE REMOVED. LEAVING IT UP COURSE WORK CHECKING**
- [notes.md](./notes.md) : This file just contains some notes and references used throughout the course of this project. **TO BE REMOVED. LEAVING IT UP COURSE WORK CHECKING**

The individual modules in the [\`src/traditional.py\`](./src/traditional.py) and [\`src/infer.py\`](./src/infer.py) files can also be imported into other Python projects using: 

`from traditional import WienerFilter, SpectralSubtraction`

or,
`from infer import DeepDenoise`


The Wiener filter and spectral subtraction are traditional methods for denoising speech signals. The Wiener filter uses a mathematical model of the noise and the clean signal to estimate the clean signal from the noisy signal. Spectral subtraction, on the other hand, estimates the noise spectrum from the noisy signal and subtracts it from the noisy signal to produce an estimate of the clean signal.

The machine learning model in the infer.py module uses a deep learning approach to denoise speech signals.

# Deep Learning Model Implementation
As mentioned in the project description, the machine learning model used by the `infer.py` module is implemented based on the paper [*"Speech Denoising with Deep Feature Losses"*](https://doi.org/10.48550/arXiv.1806.10522) by Francois G. Germain et al.
The paper presents a deep learning approach to denoising speech signals by directly processing the raw waveform. The system is trained using a "fully-convolutional context aggregation network" and a "deep feature loss", which compares the internal feature activations in a different network trained for acoustic environment detection and domestic audio tagging. The approach proposed in the paper outperforms state-of-the-art methods in objective speech quality metrics and large-scale human listener experiments. It claims to be particularly effective at denoising audio with the most intrusive background noise.
## Context Aggregation Network
This network consists of 16 convolutional layers, which are used to compute the content of each layer from the previous layer. The network is trained to handle the audio files end, with approximately a 1 second receptive field (ie how much past-temporal context the model has has when making predictions of wether or not the current sample is noise or speech). This allows the system to capture contextual information on the time scale of spoken words. The network uses dilated convolutions and adaptive normalization to aggregate long-range contextual information without changing the sampling frequency across layers. The authors indicate they expect the system to capture context on the time scales of spoken words.

### Defining the network in tensorflow:
First we follow the paper and define the network parameters and shape the input to the fit the Conv2D layers:
```python
n_layers=13 # num of internal layers
n_channels=64 # number of feature maps
# inputs are single channel waveforms shape=(N, 1)
model_input = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32)  
# additional context feature dimension (batch, time, 1, features)
inputs = tf.expand_dims(model_input, axis=-1) 
# transpose to (batch, 1, time, features)
inputs = tf.transpose(input, [0, 2, 1, 3])
```
Then we define the activation function and the adaptive normalization layer described in the paper. The adaptive normalization operator used in the proposed network is a combination of batch normalization and identity mapping of the input. This operator is used to improve the performance and training speed of the network. The weights for the normalization operator are also learned parameters. This allows the network to adapt to the specific characteristics of the input data and improve the accuracy of the denoising process.
```python
# Leaky ReLU activation function described in paper.
def LeakyReLU(x):
    return tf.maximum(0.2*x,x)
class AdaptiveNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdaptiveNormalization, self).__init__(**kwargs)
        self.alpha = tf.Variable(1.0, name='alpha')
        self.beta = tf.Variable(0.0, name='beta')
        self.batch_norm = tf.keras.layers.BatchNormalization(**kwargs)
    def call(self, x):
        return self.alpha * x + self.beta * self.batch_norm(x)
```
Then we define heart of the context aggregation network (the convolutional layers). 

    "The content of each intermediate layer is computed from the previous layer via a dilated convolution with 3 × 1 convolutional kernels [26] followed by an adaptive normalization (see below) and a pointwise nonlinear leaky rectified linear unit (LReLU) max(0.2x, x)."

    “Here, we increase the dilation factor exponentially with depth from 2^0 for the 1st intermediate layer to 2^12 for the 13th one... We do not use dilation for the 14th and last one. For the output layer, we use a linear transformation (1 × 1 convolution plus bias with no normalization and no nonlinearity) to synthesize the sample of the output signal."

```python
for current_layer in range(n_layers):
    if current_layer == 0:
        net = tf.keras.layers.Conv2D(
            n_channels, 
            kernel_size=[1, 3], 
            activation=LeakyReLU, 
            padding='SAME')(inputs)
        net = AdaptiveNormalization()(net)
    else:
        dilation_factor = 2 ** current_layer
        net, pad_elements = signal_to_dilated(
            net,
            n_channels=n_channels,
            dilation=dilation_factor)
        net = tf.keras.layers.Conv2D(
            n_channels, 
            kernel_size=[1, 3], 
            activation=LeakyReLU,
            padding='SAME')(net)
        net = AdaptiveNormalization()(net)
        net = dilated_to_signal(
            net, 
            n_channels=n_channels, 
            pad_elements=pad_elements)
net = tf.keras.layers.Conv2D(
    n_channels,
    kernel_size=[1, 3],
    activation=LeakyReLU, 
    padding='SAME')(net)
net = AdaptiveNormalization()(net)
net = tf.keras.layers.Conv2D(
    1, 
    kernel_size=[1, 1],
    activation='tanh',
    padding='SAME')(net)
# undo the transpose and squeeze the added feature dimension
output = tf.squeeze(tf.transpose(net, [0, 2, 1, 3]), axis=-1)
model = keras.Model(inputs=model_input, outputs=output)
```
Next we follow the papers suggestion for training the network. In the paper network is trained and tested using a variety of different losses, ultimately deciding in using another network to calculate deep feature loss. Additional they use L1 and L2 loss functions which is what we are using as while we are still implementing the Deep Feature Loss network. The authors suggest an "Adam" optimizer with a learning rate of `1e-4` which we replaced with a learning rate schedule that decays the learning rate by 5% every 10,000 steps. We also use the `tf.keras.utils.get_custom_objects()` to register the L1 and L2 loss functions so that we can use them in the model compile step.
```python
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000, 
        decay_rate=0.95,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    def L1_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_pred - y_true)) 
    def L2_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true)) 
    custom_objects = tf.keras.utils.get_custom_objects()
    custom_objects['L1_loss'] = L1_loss
    custom_objects['L2_loss'] = L2_loss
    model.compile(loss="L1_loss", optimizer=optimizer, metrics=['mse', 'mae', 'accuracy' ])
```
Implementation for sourcing and manipulating the data has been omitted for brevity. If you would like to train the network email me so i can run you through it.
## Deep Feature Loss Network (WIP)
For the feature loss network they authors designed a convolutional neural network inspired by the VGG architecture in computer vision. The network consists of 15 convolutional layers with 3x1 kernels, batch normalization, and LReLU activation units (very similar to the context aggregation network). Each layer is decimated by a factor of 2, halving the length of the subsequent layer compared to the preceding one. The number of channels is doubled every 5 layers, with 32 channels in the first intermediate layer.

The network is trained using backpropagation, with the output vector of the CNN serving as features for one or more logistic classifiers with a cross-entropy loss for classification tasks.(used to train the network for use in the denoising network later). 

A denoising loss function for the final network (Context Aggregation + Features Loss) is also defined, based on the: L1 loss, the difference between the feature activations of the clean reference signal (ie running the clean signal through the pre-trained feature loss network) and the output of the denoising network(ie the context aggregation module). The weights in the loss function are set to balance the contribution of each layer, and are determined by the relative values of the L1 loss after 10 training epochs. 

### Defining the Feature Loss Network
Again using tensor, you find the implementation to be very similar to the context aggregation network. In that they are both fully convolutional networks. The key difference being the lack of pooling. 
```python
# TODO
# still need to train the network
# still need to figure out how to connect the feature loss network to the main network
n_layers=14
base_channels=32
doubling_rate=5
conv_layers = []
# input 4D tensor
model_input = tf.keras.layers.Input(shape=(None, 1, None, None), dtype=tf.float32) 
# Similar structure to the other network
for current_layer in range(n_layers):
    # The number of channels is doubled every 5 layers
    # 32 channels in the first intermediate layer. 
    n_channels = base_channels * (2 ** (current_layer // doubling_rate))
    if current_layer == 0:
        # "Each Layer is decimated by 2"
        #  Just means "stride" of 2 in the time dimension.
        net = Conv2D(
            n_channels, 
            kernel_size=[1, 3],
            activation=LeakyReLU,
            stride=[1, 2],
            padding='SAME')(model_input)
        net = layers.BatchNormalization(net)
        conv_layers.append(net)
    elif current_layer < n_layers - 1:
        net = layers.Conv2D(
            n_channels, 
            kernel_size=[1, 3], 
            activation=LeakyReLU,
            stride=[1, 2], 
            padding='SAME')(conv_layers[-1])
        net = layers.BatchNormalization(net)
        conv_layers.append(net)
    else:
        net = layers.Conv2D(
            n_channels,
            kernel_size=[1, 3],
            activation=LeakyReLU,
            padding='SAME')(conv_layers[-1])
        net = layers.BatchNormalization(net)
        conv_layers.append(net)
# TODO
# "Each channel in the last layer is averaged-pooled to produce the output feature vector."

# TODO
# The logistic classifier, which is a component of the network that is used to make predictions about the audio data, is trained specifically for each individual task. This allows the network to learn task-specific information and improve its performance on each task.
# Then need to train the model on 2 separate tasks
# Inner CNN layers (n_layers) w/ different output layers
# Assuming this will be either a softmax or sigmoid layer depending on the task

# e.g:
# output1 = keras.layers.Dense(...)(x)
# output1 = keras.layers.Activation('softmax', name='output1')(output1)
# 
# output2 = keras.layers.Dense(...)(x)
# output2 = keras.layers.Activation('softmax', name='output2')(output2)

# # Compile the model at this point.
# model.compile(
#     optimizer=...,
#     loss={'output1': ..., 'output2': ...},
#     metrics={'output1': ..., 'output2': ...},
# )

# Train the model on the data for each task using binary_crossentropy loss and the Adam optimizer as mentioned in the paper.
```