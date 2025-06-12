<!-- README.md -->
# Speech Denoising Research Implementation - University Learning Project
<!-- name: "Speech Denoising Research Implementation" -->
## Project Overview
<!-- PROJECT ```
**Step 3: Dilated Convolution Layers**
The core of the network - implementing exponentially increasing dilation factors was complex:

> *"The content of each intermediate layer is computed from the previous layer via a dilated convolution with 3 × 1 convolutional kernels [26] followed by an adaptive normalization and a pointwise nonlinear leaky rectified linear unit (LReLU) max(0.2x, x)."*

> *"Here, we increase the dilation factor exponentially with depth from 2^0 for the 1st intermediate layer to 2^12 for the 13th one... We do not use dilation for the 14th and last one."*

```python
# My implementation - this was the most challenging part
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
        # These helper functions were my attempt to handle dilation properly
        # (Implementation was incomplete - major learning gap here)
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

# Final layers
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

# Output processing
output = tf.squeeze(tf.transpose(net, [0, 2, 1, 3]), axis=-1)
model = tf.keras.Model(inputs=model_input, outputs=output)
```

**My Implementation Challenges:**
- The `signal_to_dilated` and `dilated_to_signal` functions were never fully implemented
- Understanding proper dilation handling in TensorFlow was difficult
- Memory management with large receptive fields was problematicION -->
This is a **university learning project** where I explored speech denoising techniques by implementing academic research from scratch. The project was an educational exercise in understanding and implementing machine learning models for audio processing.

**What I Built:**
- A basic CLI tool that processes `.wav` files using three different denoising approaches:
  1. Spectral Subtraction (traditional method)
  2. Wiener Filter (traditional method)  
  3. Deep Learning Model (research paper implementation)

**What I Learned:**
- How to implement academic research papers in TensorFlow
- Traditional signal processing techniques for audio denoising
- Deep learning model architecture design and training
- Audio data preprocessing and manipulation with librosa
- The challenges of reproducing research results

**Honest Assessment:**
- The CLI interface was quite rough and basic
- Results were subpar compared to modern solutions
- No visual interface or demonstrations were created
- The deep feature loss network was never fully completed
- This was a learning exercise, not a production-ready tool

The traditional algorithms are based on: [*"Speech Enhancement: Theory and Practice"*](https://doi.org/10.1201/b14529) by Philipos C. Loizou.

The deep learning approach attempts to implement: [*"Speech Denoising with Deep Feature Losses"*](https://doi.org/10.48550/arXiv.1806.10522) by Francois G. Germain et al.

<!-- Mention the machine learning is implemented based on the paper  "Germain, Francois G., et al. Speech Denoising with Deep Feature Losses. arXiv:1806.10522, arXiv, 14 Sept. 2018. arXiv.org, https://doi.org/10.48550/arXiv.1806.10522." -->
<!-- And the other two algorithms come from techniques curated in "Loizou, Philipos C. Speech Enhancement: Theory and Practice. 2nd ed., CRC Press, 2013. DOI.org (Crossref), https://doi.org/10.1201/b14529." -->

## Setup and Dependencies

**Note:** This was a university project with basic setup - no automated installation or package management was implemented.

Requirements:
- Python 3.9
- TensorFlow 2.10
- librosa 0.9.23
- soundfile (latest version)
- numpy (latest version)

Manual installation required for all dependencies.

## Basic CLI Usage (Educational Purposes)

The project includes a rough command-line interface primarily built for testing and experimentation during development. 

**Basic syntax:**
```bash
python cm.py -f <input_file.wav> -a <algorithm>
```

**Arguments:**
- `-f` : Path to input `.wav` file (relative or absolute)
- `-a` : Algorithm selection:
  - `"SS"` - Spectral Subtraction
  - `"WF"` - Wiener Filter  
  - `"ML"` - Machine Learning Model

**Example:**
```bash
python cm.py -f noisy_speech.wav -a WF
```

**Output:** Creates a new file with `_<algorithm>_denoised` suffix in the same directory.

**Limitations:**
- Very basic CLI with minimal error handling
- No progress indicators or user feedback
- No audio quality metrics provided
- Results varied significantly across different audio samples
- No output filename customization (was planned but never implemented)




## Project Structure and Learning Outcomes

This section documents the different components I built while learning about audio processing and machine learning implementation.

**Core Implementation Files:**
- [`src/cm.py`](./src/cm.py) - Basic CLI entry point (rough implementation)
- [`src/traditional.py`](./src/traditional.py) - Traditional signal processing methods (Wiener Filter & Spectral Subtraction)
- [`src/infer.py`](./src/infer.py) - TensorFlow model loading and inference
- [`src/models/DenoiseFeatureLossModels.ipynb`](./src/models/DenoiseFeatureLossModels.ipynb) - Jupyter notebook for model training and experimentation

**Development/Learning Files:**
- [`src/saved/models/`](./src/saved/models) - Various experimental model checkpoints saved during training
- [`src/scripts/`](./src/scripts/) & [`deprecated/`](./deprecated/) - Testing scripts and experimental code written throughout the learning process
- [`notes.md`](./notes.md) - Personal notes and references gathered during research

**What I Learned About Each Component:**

**Traditional Methods:** Implemented mathematical approaches from academic literature, learning how classical signal processing tackles noise reduction through frequency domain manipulation.

**Deep Learning Implementation:** Gained hands-on experience translating a research paper into working TensorFlow code, understanding the challenges of reproducing academic results.

**Audio Processing Pipeline:** Learned to work with librosa for audio preprocessing, dealing with different sampling rates, and managing audio data formats.

The traditional algorithms were valuable for understanding fundamental concepts, while the ML implementation taught me about the complexities of research reproduction and model training.

## Understanding the Algorithms (Learning Summary)

**Traditional Signal Processing Methods:**

The Wiener filter and spectral subtraction represent classical approaches to speech enhancement that I implemented to understand fundamental concepts:

- **Wiener Filter:** Uses statistical modeling to estimate clean signals from noisy observations. I learned how it balances noise reduction with signal preservation through mathematical optimization.

- **Spectral Subtraction:** Works by estimating the noise spectrum and subtracting it from the noisy signal. This taught me about frequency domain processing and the challenges of avoiding over-subtraction artifacts.

These implementations helped me understand the mathematical foundations before moving to more complex deep learning approaches.

**Machine Learning Approach:**

The neural network implementation was my attempt to reproduce cutting-edge research, teaching me about the gap between paper descriptions and working code.

# My Implementation of "Speech Denoising with Deep Feature Losses"

**Learning Objective:** This section documents my attempt to implement the research paper by Francois G. Germain et al. It was a significant learning experience in translating academic research into working code.

## What I Implemented vs. What I Learned

**Successfully Implemented:**
- Context Aggregation Network architecture 
- Dilated convolutions with exponential dilation factors
- Custom adaptive normalization layers
- Basic training loop with L1/L2 losses

**Partially Implemented/Incomplete:**
- Deep Feature Loss Network (architecture defined but not fully trained)
- Complete end-to-end training pipeline
- Data preprocessing and augmentation pipeline
- Model evaluation and comparison metrics

**Key Learning Outcomes:**
- Hands-on experience implementing complex neural architectures
- Understanding the challenges of reproducing research results
- Working with TensorFlow's lower-level APIs for custom layers
- Audio processing with raw waveforms vs. spectrograms
## Context Aggregation Network - Implementation Notes

**Paper Summary:** The network uses 16 convolutional layers with dilated convolutions to capture temporal context (approximately 1 second receptive field) for distinguishing speech from noise.

**My Implementation Experience:**
This was my first time implementing dilated convolutions and custom normalization layers. The challenge was understanding how the paper's mathematical descriptions translated to TensorFlow operations.

### TensorFlow Implementation (Learning Process):

**Step 1: Input Processing**
Learning to reshape audio data for 2D convolutions was tricky - the paper wasn't explicit about tensor shapes:
```python
# What I learned about tensor manipulation for audio
n_layers=13 # num of internal layers (paper had 16, I used 13 due to computational limits)
n_channels=64 # number of feature maps
# inputs are single channel waveforms shape=(N, 1)
model_input = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32)  
# additional context feature dimension (batch, time, 1, features)
inputs = tf.expand_dims(model_input, axis=-1) 
# transpose to (batch, 1, time, features)
inputs = tf.transpose(inputs, [0, 2, 1, 3])  # Fixed variable name bug here
```

**Step 2: Custom Layers**
Implementing the adaptive normalization was educational - combining batch norm with learnable parameters:
```python
# My implementation of the paper's "adaptive normalization"
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
**Step 4: Training Configuration**
Learning about training setup and hyperparameter choices from the paper:

```python
# My training setup - learning from paper's suggestions
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,  # Started higher than paper's 1e-4
    decay_steps=10000, 
    decay_rate=0.95,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Custom loss functions as mentioned in paper
def L1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true)) 

def L2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true)) 

# Learning to register custom objects in TensorFlow
custom_objects = tf.keras.utils.get_custom_objects()
custom_objects['L1_loss'] = L1_loss
custom_objects['L2_loss'] = L2_loss

model.compile(
    loss="L1_loss", 
    optimizer=optimizer, 
    metrics=['mse', 'mae', 'accuracy']
)
```

**My Training Challenges:**
- Data pipeline implementation was never completed properly
- Limited computational resources affected model size and training time
- Difficulty reproducing the paper's training data preprocessing steps
- Never achieved the paper's reported performance metrics
## Deep Feature Loss Network - Incomplete Implementation

**Learning Goal:** This was meant to be the second part of the paper's approach - a separate network trained for audio classification that would provide "deep feature loss" signals to improve the main denoising network.

**What I Understood from the Paper:**
The authors designed a VGG-inspired CNN with 15 convolutional layers specifically for audio classification tasks. This network would be pre-trained on acoustic environment detection and domestic audio tagging, then its internal feature representations would be used to guide the training of the Context Aggregation Network.

**My Implementation Status: Incomplete**
I defined the architecture but never successfully completed the training pipeline or integration with the main network. This was a significant learning gap that taught me about the complexity of multi-network training systems.

**Why It Remained Incomplete:**
- Limited understanding of how to properly train multi-task networks
- Insufficient computational resources for training two separate networks
- Difficulty obtaining the specific audio classification datasets mentioned in the paper
- Complexity of implementing the feature matching loss function 

### My Attempted Architecture Implementation

This code represents my understanding of the paper's architecture description, but it was never successfully trained:

```python
# INCOMPLETE IMPLEMENTATION - Learning artifact
# Challenges: Understanding proper network connection, training pipeline

n_layers = 14  # Paper mentioned 15, but I used 14 in my attempt
base_channels = 32
doubling_rate = 5
conv_layers = []

# Input processing - this part I struggled with
model_input = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32)  # Simplified from paper

for current_layer in range(n_layers):
    # Channel doubling every 5 layers as mentioned in paper
    n_channels = base_channels * (2 ** (current_layer // doubling_rate))
    
    if current_layer == 0:
        # "Decimated by factor of 2" - I interpreted as stride=2
        net = tf.keras.layers.Conv2D(
            n_channels, 
            kernel_size=[1, 3],
            strides=[1, 2],  # This was my interpretation - likely incorrect
            activation=LeakyReLU,
            padding='SAME')(model_input)
        net = tf.keras.layers.BatchNormalization()(net)
        conv_layers.append(net)
    elif current_layer < n_layers - 1:
        net = tf.keras.layers.Conv2D(
            n_channels, 
            kernel_size=[1, 3], 
            strides=[1, 2],
            activation=LeakyReLU,
            padding='SAME')(conv_layers[-1])
        net = tf.keras.layers.BatchNormalization()(net)
        conv_layers.append(net)
    else:
        # Final layer without decimation
        net = tf.keras.layers.Conv2D(
            n_channels,
            kernel_size=[1, 3],
            activation=LeakyReLU,
            padding='SAME')(conv_layers[-1])
        net = tf.keras.layers.BatchNormalization()(net)
        conv_layers.append(net)

# TODO - Never implemented:
# - Proper pooling layer ("Each channel in the last layer is averaged-pooled")
# - Multi-task classification heads for different audio tasks
# - Integration with main denoising network
# - Training pipeline for classification tasks
```

**Major Learning Gaps:**
- Never figured out how to properly connect this to the main network
- Didn't implement the multi-task classification training
- Never understood how to compute the "deep feature loss" 
- Architecture choices were largely guesswork due to paper's brevity

## Project Reflection and Learning Outcomes

**What This Project Taught Me:**

1. **Research Implementation is Hard:** There's a significant gap between reading a paper and implementing working code. Many implementation details are omitted from academic papers.

2. **Deep Learning Complexity:** Building custom neural architectures requires deep understanding of tensor operations, training pipelines, and debugging skills.

3. **Traditional Methods Have Value:** The classical signal processing methods (Wiener filter, spectral subtraction) were more straightforward to implement and provided reliable baselines.

4. **Data Pipeline Challenges:** Much of the difficulty in ML projects is in data preprocessing, augmentation, and pipeline management - not just model architecture.

**If I Were to Redo This Project:**
- Start with simpler, well-documented architectures
- Focus on getting one approach working well rather than attempting three
- Spend more time on data pipeline and evaluation metrics
- Consider using existing libraries (like TensorFlow Hub) for baseline models

**Value as a Learning Exercise:**
Despite the incomplete implementation and subpar results, this project provided invaluable hands-on experience with research reproduction, TensorFlow implementation, and the practical challenges of ML development.