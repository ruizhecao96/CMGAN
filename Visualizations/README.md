# Visualization
A visualization of the CMGAN in comparison to other state-of-the-art methods is presented in the following figures.
The methods were chosen based on the availability of open-source implementations and the reproducibility
of the reported results in the corresponding papers. As a representative for metric discriminator, we used the
MetricGAN+ [1]. For time-domain methods, DEMUCS [2] is selected. For TF-domain complex denoising,
PHASEN [3] is chosen as it attempts to correct magnitude and phase components. In addition to, PFPL
[4] utilizing a deep complex-valued network to enhance both real and imaginary components. Most of the
papers provided an official implementation with pretrained models. PHASEN is the only exception, as a
non-official code is used and we trained the model to reproduce the results in the paper. For DEMUCS,
the available model is pretrained on both Voice Bank+DEMAND and DNS challenge data. Thus, we
retrain DEMUCS using the recommended configuration on Voice Bank+DEMAND data only to ensure a
fair comparison between all presented models.<br>

A wide-band non-stationary cafe noise from the DEMAND dataset (SNR = 0 dB) [5] and a narrowband
high-frequency stationary doorbell noise from the [Freesound](https://freesound.org/) dataset (SNR = 3 dB) are used to
evaluate the methods. Both noises are added to sentences from the DNS challenge [6]. Comparisons are
made between time-domain, TF-magnitude, and TF-phase representations for comprehensive performance
analysis. Since the phase is unstructured, we utilize the baseband phase difference (BPD) approach
proposed in [7] to enhance the phase visualization. From Fig. 1, MetricGAN+, DEMUCS and PHASEN
show the worst performance by confusing speech with noise, particularly in the 1.5 to 2 seconds interval
(similar speech and noise powers). The distortions and missing speech segments are annotated in the time
and TF-magnitude representations by the blue and red arrows, respectively. Moreover, the denoised phase in methods
employing only magnitude (MetricGAN+) and time-domain (DEMUCS) is very similar to the noisy input,
in contrast to clear enhancement in complex TF-domain methods (PHASEN, PFPL and CMGAN). PFPL
and CMGAN exhibit the best performance, with better phase reconstruction in CMGAN.<br>

In general, stationary noises are less challenging than non-stationary counterparts. However, stationary
noises are not represented in the training data. As depicted in Fig. 2, methods such as MetricGAN+
and PHASEN are showing a poor generalization performance, with doorbell distortions clearly visible
at frequencies (3.5, 5, and 7 kHz). On the other hand, the performance is slightly better in DEMUCS
and PFPL, whereas CMGAN perfectly attenuates all distortions. Note that high-frequency distortions are
harder to spot in the time-domain than in TF-magnitude and TF-phase representations. [Audio samples](https://sherifabdulatif.github.io/) 
from all subjective evaluation methods are available for interested readers.

<img src="https://github.com/ruizhecao96/CMGAN/blob/main/Visualizations/wb_noise_svg_w.png" width="1200px"> <br><br>

<img src="https://github.com/ruizhecao96/CMGAN/blob/main/Visualizations/nb_noise_svg_w.png" width="1200px">

[1] S. W. Fu et al., “MetricGAN+: An improved version of MetricGAN for speech enhancement,” in Proc. Interspeech, 2021, pp. 201–205.<br>
[2] A. Defossez, G. Synnaeve and Y. Adi, “Real time speech enhancement in the waveform domain,” in Proc. Interspeech, 2020, pp.
3291–3295.<br>
[3] D. Yin et al., “PHASEN: A phase-and-harmonics-aware speech enhancement network,” in Proc. of the Conference on Artificial
Intelligence, vol. 34, no. 05, 2020, pp. 9458–9465.<br>
[4] T. A. Hsieh et al., “Improving perceptual quality by phone-fortified perceptual loss using wasserstein distance for speech enhancement,”
arXiv, vol. abs/2010.15174, 2020.<br>
[5] J. Thiemann, N. Ito and E. Vincent, “The diverse environments multi-channel acoustic noise database (DEMAND): A database of
multichannel environmental noise recordings,” in Proc. of Meetings on Acoustics, vol. 19, no. 1, Acoustical Society of America, 2013.<br>
[6] H. Dubey et al., “ICASSP 2022 Deep noise suppression challenge,” in IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), 2022.<br>
[7] M. Krawczyk and T. Gerkmann, “STFT phase reconstruction in voiced speech for an improved single-channel speech enhancement,”
IEEE Transactions on Audio, Speech, and Language Processing, vol. 22, no. 12, pp. 1931–1940, 2014.<br>
