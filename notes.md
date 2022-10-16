# Datasets
Currently using the following datasets:
- [LibriSpeech ASR corpus](http://www.openslr.org/12).
    - currently only using the `dev-clean` dataset.

Look into:
- [VTCK]() for a higher fidelity dataset.
- [UrbanSound8K] for noise samples
# Background
## Timeline
### 2016
-   [Complex Ratio Masking (CRM)](https://ieeexplore.ieee.org/document/7472673) | 03/2016
    - works in complex domain
    - jointly enhance both magnitude and phase
    - uses DNN to estimate the complex components of the mask ('ideal' is what they call it, C'I'RM).
        - able to estimate both magnitude and the phase by operating in complex domain
        - goal of the mask is that when applied to the STFT of the noisy speech, the resulting STFT should be as close as possible to the STFT of the clean speech.
        - predicting complex components of the magnitude spectra may not be as good as predicting the components of the complex ideal ratio mask.
    - works better than separately enhancing magnitude and phase.
    - cIRM is closely related to the wiener filter. which is a complex ratio of the cross-power spectrum for the clean and noisy signals.
    - DNN has 3 hidden layers, each with 1024 units and ReLU functions. MSE cost function.
        - output split into 2 layers, one for real and one for imaginary components
        - input is features from STFT of noisy speech
            - AMS (amplitude modulation spectrogram)
            - MFCC (mel-frequency cepstral coefficients)
            - RASTA-PLP (relative spectral transform and perceptual linear prediction)
-   [Fully Convolutional Neural Networks (FNN) for Speech Enhancement](http://arxiv.org/abs/1609.07132) | 09/2016
    - uses a FNN or FCNN to find an encoder 'mapping' between noisy and clean signal spectra.
        - CNN's consist of less parameters than any Fully connected network. This is because of the weight sharing property 
        - for this task FNN's are smaller and faster than something like a RNN
    - They call their network a "Redundant Convolutional Encoder-Decoder" Network (R-CED).
        - no pooling layers
        - encodes the features into higher dimensions along the encoder and compresses back down along the decoder.
        - num of filters kept symmetric, increasing and decreasing along the encoder and decoder.
        - a 'Cascaded' variant (CR-CED)consist of repetitions of R-CEDs, it achieves better performance with faster convergence.
        - CED's utilize 'skip' or 'residual' connections every other layer starting from the 2nd layer. 
            - skip connections are usually useful for vanishing gradient problems
        - uses a 1D Conv 
    - main concern is with small size that can be used for real-time embedded applications like hearing aids.
    - extract redundant representations of any noisy spectrum at the encoder and then map it back to a clean spectrum at the decoder.
        - spectrums are basically mapped to a higher dimensional space via encoding and projected back to the original space.
    - FCNSS can model temporal atributes of time series data.
### 2018
- [Waveform metric optimization using FCNN's](http://arxiv.org/abs/1709.03658)
    - proposes using STOI (short-time objective intelligibility) as a loss function for training a FCNN, in the task domain of human speech, directly instead of using MMSE (mean squared error).
    - FCNN's are like CNN's with no fully connected layers of any kind. (no pooling or dense layers)
        - since it only consists of convolution layers the local features are better preserved with less weights required.
    - Traditional DNN's and CNN's can only process fixed length inputs (because of the fully connected layers), but using waveforms as inputs and FCNN's can process any length input.
    - considering both STOI and MMSE gives better ASR performance than using just MMSE.
    - this study assumed that most DNN methods were not using the complex domain and claimed that phase was not considered in most DNN methods (we know this is not true now).
    - since its waveform in/waveform out it doesnt deal with the computational load of mapping back and forth between waveform and time-frequency (TF) domain like when obtaining STFTs.
    - convolving a  time domain signal with a filter is equivalent to multiplying the frequency response of the filter with the frequency of the original signal at that point.
    

### 2019
-   [Deep Filtering, signal extraction and reconstruction using complex TF features](http://arxiv.org/abs/1904.08369)
    - uses the process of finding a CRM 
    - estimates a complex TF filter for each mixture (noise + clean signal) TF bin.
    - DNN is optimized by minimizing error between the extracted and ground truth signals. 
        - lets you learn the filters without having to know the ground truth filters.
    - the mixture STFT is also processed with notch-filters, and 0 whole time-frames (simulates packet loss).
        - this allows for showing the reconstruction capabilities.
    - Goal is to extract and reconstruct the clean signal from the noisy signal in a joint manner.
    - Model finds complex value "filters" as opposed to "masks" (as with CRM).
        - 1 filter for each stft bin
        - filters are element-wise applied a defined area in the stft mixture.
            - results are summed to get the output signal. 
        - Each estimated TF bin is a complex weighted sum of a TF bin area in the complex mixture, this is why the model has reconstructive properties.
            - where a TF mask lets you use context from only one TF bin to estimate a target TF bin, the filter lets you use context from multiple TF bins in an area around the target TF bin.

        - the filter is referred to as the "Deep Filter" (DF)
        - complex conjugate 2D filter of TF bin (n, k)
    - Thier approach of DNN + TF filters mitigates both the cons of TF masks or TF filters.
    - As part of their tests they also used mask estimation.
        - DNN with 3 BLSTM, 1200 unit per, FF output layer.
    - Trains against clean target signals.
### 2021
- [Noisy2Noisy, deep speech denoising](https://linkinghub.elsevier.com/retrieve/pii/S0003682X20307350)
    - deep learning model that "doesnt require any clean speech data", so it can be trained on noisy data in a supervised manner.
    - it actually builds off of models trained in a supervised fashion, as would classiclaly be done for the task
        - it has to train a DNN first prett much, this DNN can be used in the self-supervised process of training.
    - while training in unsupervised fashion, it uses a noised signal as both the input and output of the network (noisy2noisy).
        - durring training if clean speech target signals are replaced with noisy signals with expected values equal the clean signal the final weight of the network will be the same.
            - provided both signals have 0 mean noise, and both signals are uncorrelated
    - the supervised DNN is considered a FCNN or FNN (fully convolutional NN).
    - also uses a VAD to distringuish between speech and 'silent' segments classify the noise.
        - separate FCNNs can be trained for each noise class.
        - proper FCNN is then chosen by a classifier inside of the supervised model.
    - if an unkonwn noise class is encountered, the supervised model can create a new bank and train it on the new noise class.

-   [DiffWave, a diffusion model for audio synthesis](http://arxiv.org/abs/2009.09761) | 03/2021
    - proposes a probabilistic model for audio synthesis.
        - non auto-regressive, converts white noise  into a structured audio waveform via a markov chain.
    - DiffWave is comparable to WaveNet (auto-regressive) in terms of speech quality for something like a vocoding task, but is orders of magnitude faster.
        - outperforms auto-regressive and GAN models in unconditional generation.
        - DiffWave uses a feed-forward and bidirectional CNN motivated by WaveNet.
    - markov chain gradually converts a simple noise distribution into a complicated data distribution (target audio).
    - diffusion is a step wise noise adding process where a "whitened" latent (some white noise signal) is generated iteratively from the training signal.
    - the non auto-regressive nature of the model allows for parallelization.
    - can be conditioned to generate conditional waveforms using different features sets
        - aligned linguistic features
        - mel-spectrograms from text-to-spectrogram models
        - hidden states from text-to-wave architectures
    - the dialated convolutional layers are used to model the temporal structure of the audio signal.
        - increasing the number of layers or size of cycles can lead to degradation in quality in something like WaveNet but in DiffWave it leads to better quality because it has a larger receptive field.
    
    - has positionally large receptive field
    - Has some of the best performance for smaller utterances in unconditional and class-conditional generation tasks,
    - DiffWave is still slower than flow-based models like Glow and RealNVP. 
### 2022
- [DeepFilterNet](http://arxiv.org/abs/2110.05588) | 02/2022
    - 2 Stage speech enhancement system
        - 1st stage enhances the spectral envelope using ERB-scaled gains
            - ERB helps model human hearing.
            - ERB reduces the input and output dimensions to only 32 bands.
        - 2nd stage uses the process of Deep Filtering to find a TF filter and enhance the periodic components.
            - DF is only applied on the lower frequencies since the ERB misses those.
            - periodic compnents contain most of their energy in the lower frequencies. 
            - DF does not provide any benefit for the higher frequencies over ERB gains in noise only sections.
                - DF can even cause artifacts trying to model noise only sections. 
    - low complexity architecture
        - separable convolutions
        - grouping in linear and recurrent layers.
    - DF shown to be superior to CRM for multiple FFT sizes (5-30ms)
    - adopts a U-Net like architecture. 
        - convolution recurrent U-net 
    - uses skip connections in the encoder decoder called "pathway convolutions"
        - also uses a skip from the GGRU to the DF Net

    
    

