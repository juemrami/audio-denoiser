# Datasets
Currently using the following datasets:
- [LibriSpeech ASR corpus](http://www.openslr.org/12).
    - currently only using the `dev-clean` dataset.
- [NOIZEUS](https://ecs.utdallas.edu/loizou/speech/noizeus/)
- [UrbanSound8K](https://urbansounddataset.weebly.com/download-urbansound8k.html) for noise samples
- [Noisy speech database for training speech enhancement algorithms and TTS models](https://datashare.ed.ac.uk/handle/10283/2791)
    - think im using the clean only
- 
Look into:
- [VTCK]() for a higher fidelity dataset.
- [QUT-NOISE](https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/)

General Notes:
- convolutions
    - sliding window.
    - at each step, use the data from the window to compute the output.
    - in terms of audio in the time domain, the window is a set of samples
    - moving average convolution
        - output is the sum of the input samples in the window divided by the number of samples in the window
        - the moving average window is also called an impulse response
        - it is FIR do it has bumps
        - at some point the impulse response is 0
    - running average uses all previous samples as its window. 
        - weighting coefficient (alpha) is used to weight the current sample
            - also servse as the frequency for the lowpass filter
            - less than 1
        - IIR
            - approaches but never becomes 0
            - smoother magnitude frequency response
    - Convolution in the time domain is multiplication in the frequency domain 

- a lowpass filter is a smoothing operation that removes high frequency (less smooth "jagged" edges) from a signal.
- "ideal" low pass filters are like square waves. they cut off all frequencies above a certain point.
    - sounds like a binary mask
- “The most progressive algorithm of speech enhancement is the spectral subtraction based on perceptual properties. This algorithm takes advantage of how people perceive the frequencies instead of just working with SNR. It results in appropriate remnant noise suppression and acceptable degree of speech distortion.” ([Upadhyay and Karmakar, 2015, p. 583](zotero://select/library/items/366CBNXZ)) ([pdf](zotero://open-pdf/library/items/5VUAHX8Q?page=10&annotation=CRM5XLJV))
- If you wanted to reconstruct your original signal using each of the frequency components in the FFT, you would use the phase values to time align each of the sinusoids so you get the proper destructive and constructive interference to reproduce the original signal.

- “Phase related problems disappear when sounds are just represented as magnitude or power spectrograms, since different realizations of the same sound are almost identical in this time-frequency plane” ([Lluís et al., 2019, p. 1](zotero://select/library/items/RE8C295N)) ([pdf](zotero://open-pdf/library/items/62PPECZU?page=1&annotation=YYN8VTKF))

- Power spectrum is the abs(X)^2  of a signal X in the complex tf domain. (so gets ride of the complex with the absolutve value).
    - Log power spectrum is the log of the power spectrum.

- A filter is a product of freqency domain reprentations of signals
- “It is possible to reduce the background noise, but at the expense of introducing speech distortion, which in turn may impair speech intelligibility. Hence, the main challenge in designing effective speech enhancement algorithms is to suppress noise without introducing any perceptible distortion in the signal.” ([Loizou, p. 1](zotero://select/library/items/IBKAK3L2)) ([pdf](zotero://open-pdf/library/items/YVIX94UY?page=29&annotation=25N66ZYG))

- sampling create aliases (or copies) of the original signal (centered at higher frequencies).
    - happens when changing sampling rates
    - This is why the audio sounds so whiny when converting back to time domain (istft) 
        - sample rate is increased to a point that is infitsimal (continious)
    - fix this by lowpass filtering the spectrum (stft)
    - filtering in stft domain has a delay since u need a set of samples at least
    - can anti-alias at a sample level in the time domain
        - moving average convolutions
    
- direct convoltiuion is more operations than FFT
- only FIR filters can be used in FFT domain

- 0 padding in the time domain allows you to shift a convolution operation back some samples so that the convolution has some dummy "past data" for the output points its producing and the end of a convolution signal lines up with the start of the signal to be convoluted.
    - this makes the convolution causal since each point now only requires on data that has been seen.

- 0 padding in the Fourier Transform is usually done at the END of a signal but it is used to increase the frequency resolution

- better temporal resolution would be preffered for the VAD segments, but to pick out frequencies to filter durring speech youd want better frequency resolution,

- “The STFT X(n, ω) can be interpreted in two distinct ways, depending on how we treat the time (n) and frequency (ω) variables. If, for instance, we assume that n is fixed but ω varies, then X(n, ω) can be viewed as the Fourier transform of a windowed sequence. If we assume that ω is fixed and the time index n varies, a filtering interpretation emerges” ([Loizou, p. 32](zotero://select/library/items/IBKAK3L2)) ([pdf](zotero://open-pdf/library/items/YVIX94UY?page=60&annotation=UDW4U9QA))

-The a priori SNR is the ratio of the power of the clean signal and of the noise power. The a posteriori SNR is the ratio of the squared magnitude of the observed noisy signal and the noise power. Both SNRs are computed for each frequency bin
- Turns out that convolution and correlation are closely related. For real signals (and finite energy signals) [read more](https://dsp.stackexchange.com/questions/55388/for-complex-values-why-use-complex-conjugate-in-convolution)

- https://dsp.stackexchange.com/questions/37059/are-there-any-realtime-voice-activity-detection-vad-implementations-available


**Weiner Filtering**
Filter H is the ratio of the (power spectrum of the clean signal) / (power spectrum of the clean signal + power spectrum of the distortion noise).
H is applied to the noisy signal in the frequency domain to get the clean signal.

WF are fixed gain at all frequencies, so they are not adaptive.
WF also requires you the esiimate the power spectrum of the noise, which is not always possible.
WF cannot be non-causal since speech cannot be assumed to be stationary.

the problem with a wiener filter is that it kills off signals along with the noise when both are present. (so it is not adaptive)
https://vocal.com/noise-reduction/the-simple-theory-of-noise-reduction-wiener-filtering/

“The Wiener filter gives the MMSE estimate of the short-time Fourier transform (STFT) whereas the spectral subtraction obtains the MMSE estimate of the short-time spectral magnitude without changing the phase [2-3, 6-8]” ([Upadhyay and Jaiswal, 2016, p. 26](zotero://select/library/items/GNIB4ICH)) ([pdf](zotero://open-pdf/library/items/JRZN8Y3D?page=5&annotation=WRX8RXH7))

“Thus, the Wiener filter attenuates each frequency component in proportion to the estimated SNR (ξk) of that frequency.” ([Loizou, p. 147](zotero://select/library/items/IBKAK3L2)) ([pdf](zotero://open-pdf/library/items/YVIX94UY?page=175&annotation=MJBZLXG3))

**Adaptive Weiner Filtering**
- attenuates each frequency component by a certain amount depending on the power of the noise at the frequency.
    - if the power of the noise  estimate is 0 then the gain is 1.
    - if the power of the noise estimate is power of the fed noisy signal, then the gain is 0.
    - “it can be observed that the WF is based on the ensemble average spectra of the signal and noise, whereas the SSF uses the instantaneous spectra for noise signal and the running average (time-averaged spectra) of the noise.” ([Upadhyay and Karmakar, 2015, p. 579](zotero://select/library/items/366CBNXZ)) ([pdf](zotero://open-pdf/library/items/5VUAHX8Q?page=6&annotation=2YQ43C5G))



# Background

“This binary function is often called the binary mask and in ideal conditions (where there is a priori knowledge of the clean and masker signals), it is called the ideal binary mask” ([Loizou, p. 619](zotero://select/library/items/IBKAK3L2)) ([pdf](zotero://open-pdf/library/items/YVIX94UY?page=647&annotation=ZJUMZY66))


## Timeline
### 2016
- [Single Channel Speech Enhancement: using Wiener Filtering with Recursive Noise Estimation](https://www.sciencedirect.com/science/article/pii/S1877050916300758) | 01/2016
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
-   [Fully Convolutional Neural Networks (FCN) for Speech Enhancement](http://arxiv.org/abs/1609.07132) | 09/2016
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
### 2017 
- [FCN's for Raw Waveform speech enhancement](http://arxiv.org/abs/1703.02205)
    - waveform-in waveform-out model. 
    - claim DNN and CNN based models lack in capability to restore high frequency components of speech.
    - Do no use fully connected layers because they do no preserve local information.
        - "We argue that this kind of connection produces difficulties in modeling high and low frequency components of waveform simultaneously."
        - "it is difficult to “learn” the weights in fully connected layers to generate high and low frequency parts of a waveform simultaneously."
    - FCNs have been used for modeling raw pixels outputs in CV
    - FCNS better model high and lower frequencys becauses instead of depending on weights theyre dependant on the ancestor nodes in the receptive field which are used to caclulate conv filters. These filters can vary between samples unlike weights.
    - "fully connected layers destroy the correlation between features, making it difficult to generate waveforms."
    - "that since the weight vectors in last fully connected layer are highly correlated to each other, it is difficult for them to produce high frequency waveform (as in the lower part of Fig. 7). However, if we only use one node, then the problem can be solved (as in the upper part of Fig. 7)"

### 2018
- [A Wavenet for speech denoising](10.48550/arXiv.1706.07162) | 01/2018
    - updates on Wavenet by getting ride of the autoregressiveness and using a dilated causal convolutional network.
    - Wavenet
        - Wavenet makes use of causal, dilated convolutions [30, 37]. It uses a series of small (length = 2) convolutional filters with exponentially increasing dilation factors. This results in a exponential receptive field growth with depth. Causality is enforced by asymmetric padding proportional to the dilation factor, which prevents activations from propagating back in time – see Figure 2 (Right). Each dilated convolution is contained in a residual layer [8], controlled by a sigmoidal gate with an additional 1x1 convolution and a residual connection – see Figure 2 (Left).
        - Wavenet uses a discrete softmax output to avoid making any assumption on the shape of the output’s distribution – this is suitable for modeling multi-modal distributions
    - Makes use of future context given there will be latency anyways. 
    - Causal meaning it uses previous data. so autoregressive. same thing.
    - **autoregressive**: previously generated samples are fed back into the model to inform future predictions.

    - Proposed model removed the autoregressive  & causal nature of Wavenet
    - What theyre trying to do is make the sample to be esimated be the middle of a symmetric receptive field (so using both past and future context).
        - This eleminiates causality somehow.
    - does not use STOI direct loss but instead compares
        - signal estimate and actual signal plus background signal and background estimate
    - “Note that the proposed model is no longer autoregressive and its output is not explicitly modeling a probability distribution, but rather the output itself. Furthermore, the model is trained in a supervised fashion – by minimizing a regression loss function. As a result: the proposed model is no longer generative (like Wavenet), but discriminative.” ([Rethage et al., 2018, p. 5](zotero://select/library/items/GCS6FNQR)) ([pdf](zotero://open-pdf/library/items/WUECUE46?page=5&annotation=N3I5XEGR))
    - “The relatively small size of the model (6.3 million parameters) together with its parallel inference on 1601 samples at once, results in a denoising time of ≈ 0.56 seconds per second of noisy audio on GPU.” ([Rethage et al., 2018, p. 6](zotero://select/library/items/GCS6FNQR)) ([pdf](zotero://open-pdf/library/items/WUECUE46?page=6&annotation=HNKSNC7V))
        - so its a little slow. 
- [Hybrid DSP/DL approach to SE](http://arxiv.org/abs/1709.08243) | 05/18
    - uses RNN to to estimate critical gain bands. then more traditional pitch filtering to attenuate noise between harmonics.
    - does not use cIRM but rather just an IRM computed on low res spectral envelope.
    - “Rather than rectangular bands, we use triangular bands, with the peak response being at the boundary between bands.” ([Valin, 2018, p. 2](zotero://select/library/items/SKYGF7GN)) ([pdf](zotero://open-pdf/library/items/FWCHZNGK?page=2&annotation=RTVC98ZM))
    

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
- [Source seperation in the waveform domain](http://arxiv.org/abs/1810.12187) | 06/19
    - assumes no work has been done in the frequency domain accounting for phase.
    - implement WaveNet and Wave-U-Net like models.
    - WaveNet
        - autogressive model
        - uses probabilistic modeling of the signal
            - Wavenet makes use of causal, dilated convolutions [30, 37]. It uses a series of small (length = 2) convolutional filters with exponentially increasing dilation factors. This results in a exponential receptive field growth with depth. Causality is enforced by asymmetric padding proportional to the dilation factor, which prevents activations from propagating back in time – see Figure 2 (Right). Each dilated convolution is contained in a residual layer [8], controlled by a sigmoidal gate with an additional 1x1 convolution and a residual connection – see Figure 2 (Left).


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
### 2020
- [CLCNet, noise reduction using complex linear coding](http://arxiv.org/abs/2001.10218) | 01/2020
    - aims to target speech enchancement with "low-res" spectrograms
    - is phase aware.
    - uses CLC (complex linear coding) as an extension to LPC (linear predictive coding) to estimate the clean speech signal.
        - attempts to use a NN to solve for the LPC coefficients. that minimize the error between the clean and noisy speech.
        - just calculating the coeffiencents is pretty bad though,
            - only works on frames with actual speech.
            - LPC coefficients only slightly reduce white noise.
        - "Given a noisy spectrogram, the model predicts complex valued coefficient that are applied to the noisy spectrogram again. Thus, CLC will output an enhanced spectrogram that can be transformed into time-domain” ([Schröter et al., 2020, p. 3](zotero://select/library/items/2ZCX9QD9)) ([pdf](zotero://open-pdf/library/items/D86XRUU6?page=3&annotation=HB9QGAJ2))"
        -“In contrast to LPC, for CLC we can use information of the current and even future frames resulting in a more general form of the linear combination in” ([Schröter et al., 2020, p. 3](zotero://select/library/items/2ZCX9QD9)) ([pdf](zotero://open-pdf/library/items/D86XRUU6?page=3&annotation=FX7SNR7D))
    
    - “We showed that our CLC framework is able to reduce noise within individual frequency bands while preserving the speech harmonics.” ([Schröter et al., 2020, p. 4](zotero://select/library/items/2ZCX9QD9)) ([pdf](zotero://open-pdf/library/items/D86XRUU6?page=4&annotation=2BTTH8DE))
    - 
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
- [ CRN for noise supression](http://arxiv.org/abs/2101.09249) | 05/2021
    - Frequency domain method.
    - Compare RNN and CRN models (Convolutional Recurrent Neural Networks) for noise suppression.
    - Ultimately propose an efficient CRN model with less computational operations.
    - CRUSE network
        - CRU NET for Speech Enhancement
        - Has a Symmetrical encoder / decoder architecture network.
            - this one actually does compress via downsampling (along frequency axis) and striding but adopts skips to recover the original information during the decode process. 
            - in the middle of the encoder decoder sits a reccurent block.  
                - 2 LSTMS or a Single GRU layer.
        - proposes way to parallize the middle block by using many GRU's doing work on a small part of the sample.
        - Skip layers are simply added to the input during the decode process.
            - minor performance degradation over concatenation.
            - concatenation requires double inputs at decoder.
        - To train the skip connection they implement the skip connection as a convolution 1x1 layer with channel size = to layer size at the state of the encoder/decoder. 
        

### 2022
- [TFCN: Temporal-Frequential Convolutional Network for Single-Channel SE](http://arxiv.org/abs/2201.00480) | 01/2022
    - TCN (Temporal Convolutional Network) is a popular model for time series data.
        - “The performance advantage of TCN comes from its large and flexible receptive filed and capability to learn the local and long-time context information.” ([Jia and Li, 2022, p. 1](zotero://select/library/items/WVJSYF69)) ([pdf](zotero://open-pdf/library/items/8IPERUIV?page=1&annotation=ES7V54TH))
        - TCNN is like U-Net as well. Encoder/Decoder architecture but in the middle instead of a recurrent block there is a TCN (Temporal Convolutional Network).
            - better performance than CRN (like CRUSE)
        - TCN's are usually applied in the TF-domain (stft) and convolutions along the time axis and fully connected on the frequency axis.
    - TFCN Similar to TCNN
        - TFCN replaces the 1-D convolutions in TCN with 2-D convolutions.
        - TFCN combines the large receptive field of TCN with the TF modeling ability of U-Net.
    - TFCN is non-casual which mean its uses a both past and future samples to predict the current sample.
        - It can be made casual or semi-casual by use of padding.
    - claims TCNN is full convolutional.
    - “The DNN is the core of the system, mapping the LPS of the noisy signal YLP S into the LPS of the estimated signal ˆ SLP S, which is then combined with the noisy phase Yp to generate the estimated result ˆ s.” ([Jia and Li, 2022, p. 2](zotero://select/library/items/WVJSYF69)) ([pdf](zotero://open-pdf/library/items/8IPERUIV?page=2&annotation=WQQ4XUMX))
    -
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
- [DeepFilterNet2](10.48550/arXiv.2205.05474) | (05/22)
    - extracts ERB and Complex features from an STFT of the input signal.
    - enocdes both signals.
    - 1 output will go to the ERB decoder and the other to the DF decoder.
    - ERB decoder output will be used to calculte the ERB gaines to the speech envelope 
    - the DF decoder will be used to calculate the TF filter for the periodic components. (only the lower frequencies below 5khz)



    
    

