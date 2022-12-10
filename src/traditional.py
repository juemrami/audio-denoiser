import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
def WienerFilter(filename, sr=16000, n_fft=512, hop_rate=0.5, vad_db=5, gamma=1.0, G=.6):
    x, _ = librosa.load(filename, sr=sr)

    hop = int(hop_rate*n_fft)         # hop size in samples

    X = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    # setting default parameters
    vad_db = 5       # VAD vad_dbhold in dB SNRseg
    gamma = 1.0     # exp(gamma)
    G = .6 #smoothing factor

    noise_mean = np.zeros((n_fft//2+1))
    for k in range(0, 5):
        noise_mean = noise_mean + abs(X[:, k])

    # noise estimate from first 5 frames
    noise_mu = noise_mean / 5

    # initialize various variables
    img = 1j
    X_out = np.zeros(X.shape, dtype=complex)

    # main processing loop
    for n in tqdm(range(0, X.shape[1])):
        # extract a frame
        signal_spec = X[:, n]
        # compute the magnitude
        signal_magnitude = abs(signal_spec)
        # save the noisy phase information
        theta = np.angle(signal_spec)
        #  compute segmental SNR for VAD
        SNRseg = 10 * np.log10(np.linalg.norm(signal_magnitude, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        # perform the spectral subtraction
        clean_signal_magnitude = signal_magnitude ** gamma - noise_mu ** gamma

        # halfwave rectification (zero out negative values)
        clean_signal_magnitude = np.maximum(clean_signal_magnitude, 0)

        # compute a Priori SNR (used)
        SNRpri = 10 * np.log10(np.linalg.norm(clean_signal_magnitude, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

        # parameter band dependent oversubtraction factor
        mu_max = 20
        mu_to_plus, mu_to_min = 1, mu_max
        mu_slope = ((mu_to_min - mu_to_plus) * mu_max) / 25
        mu_0 = mu_to_plus + 20*mu_slope
        def get_alpha(SNR):
            if SNR >= 20:
                 return mu_to_plus
            elif -5.0 <= SNR <= 20.0:
                return mu_0 - SNR*mu_slope
            else: return mu_to_min
        alpha = get_alpha(SNRpri) 

        # 2 gain function G
        # This is essentially the inverse Wiener Filter
        G_i = clean_signal_magnitude ** 2 / (clean_signal_magnitude ** 2 + alpha * noise_mu ** 2)        
        wf_speech = G_i * signal_magnitude

        # --- implement a simple VAD detector --- #
        if SNRseg < vad_db:  # Update noise spectrum
            noise_temp = G * noise_mu ** gamma + (1 - G) * signal_magnitude ** gamma  # noise power spectrum smoothing
            noise_mu = noise_temp ** (1 / gamma)  # New noise amplitude spectrum
            clean_signal_magnitude = .2*signal_magnitude  # suppress the signal    
        # add phase    
        phased_clean_signal = (wf_speech ** (1 / gamma)) * np.exp(img * theta)       
        # store the output
        X_out[:, n] = phased_clean_signal
        signal = librosa.istft(X_out, hop_length=hop, n_fft=n_fft)
        outfile = filename.split('.')[0] + '_wf_denoised.wav'
        sf.write(outfile, signal, sr)
def SpectralSubtraction(filename, sr=16000, n_fft=512, hop_rate=0.5, noise_thresh=2, gamma=1.0, G=1):
    hop = int(hop_rate*n_fft)# hop size in samples
    x, sr = librosa.load(filename, sr=16000)
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop)
    
    # noise estimate vector
    noise_mu = np.zeros((n_fft//2+1))
    
    # initialize various variables
    img = 1j
    X_out = np.zeros(X.shape, dtype=complex)
    # start
    for n in tqdm(range(0, X.shape[1])):
        # compute fourier transform of a frame
        signal_spec = X[:, n]
        # compute the magnitude
        signal_magnitude = abs(signal_spec)
        # save the noisy phase information
        theta = np.angle(signal_spec)
        # SNR
        SNR = 10 * np.log10(np.linalg.norm(signal_magnitude, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)
        
        # Simple spectral subtraction with only a fre params #
        clean_signal_magnitude = signal_magnitude ** gamma - noise_mu ** gamma
        
        # halfwave rectification (zero out negative values)
        clean_signal_magnitude = np.maximum(clean_signal_magnitude, 0)
    
        # --- implement a simple noise level detector --- #
        if SNR < noise_thresh:  # Update noise spectrum
            noise_temp = G * noise_mu ** gamma + (1 - G) * signal_magnitude ** gamma  # Smoothing processing noise power spectrum
            noise_mu = noise_temp ** (1 / gamma)  # New noise amplitude spectrum
            clean_signal_magnitude = .2*signal_magnitude  # suppress the signal
        
        # add phase    
        phased_clean_signal = (clean_signal_magnitude ** (1 / gamma)) * np.exp(img * theta)       
        # store the output
        X_out[:, n] = phased_clean_signal
        x_out = librosa.istft(X_out, hop_length=hop, n_fft=n_fft)
        outfile = filename.split('.')[0] + '_ss_denoised.wav'
        sf.write(outfile, x_out, sr)