# CMGAN: Conformer-Based Metric GAN for Monaural Speech Enhancement (https://arxiv.org/abs/2209.11112)

## Abstract:
Recently, convolution-augmented transformer (Conformer) has achieved promising performance in automatic speech recognition (ASR) and time-domain speech enhancement (SE), as it can capture both local and global dependencies in the speech signal. In this paper, we propose a conformer-based metric generative adversarial network (CMGAN) for SE in the time-frequency (TF) domain. In the generator, we utilize two-stage conformer blocks to aggregate all magnitude and complex spectrogram information by modeling both time and frequency dependencies. The estimation of magnitude and complex spectrogram is decoupled in the decoder stage and then jointly incorporated to reconstruct the enhanced speech. In addition, a metric discriminator is employed to further improve the quality of the enhanced estimated speech by optimizing the generator with respect to a corresponding evaluation score. Quantitative analysis on Voice Bank+DEMAND dataset indicates the capability of CMGAN in outperforming various previous models with a margin, i.e., PESQ of 3.41 and SSNR of 11.10 dB. 

[Demo of audio samples](https://sherifabdulatif.github.io/cmgan/) 

A longer detailed version is now available on [arXiv](https://arxiv.org/abs/2209.11112).

The short manuscript is published in [INTERSPEECH2022](https://www.isca-speech.org/archive/interspeech_2022/cao22_interspeech.html). 

Source code is released!

## How to train:

### Step 1:
In src:

```pip install -r requirements.txt```

### Step 2:
Download VCTK-DEMAND dataset with 16 kHz, change the dataset dir:
```
-VCTK-DEMAND/
  -train/
    -noisy/
    -clean/
  -test/
    -noisy/
    -clean/
```

### Step 3:
If you want to train the model, run train.py
```
python3 train.py --data_dir <dir to VCTK-DEMAND dataset>
```

### Step 4:
Evaluation with the best ckpt:
```
python3 evaluation.py --test_dir <dir to VCTK-DEMAND/test> --model_path <path to the best ckpt>
```

## Model and Comparison:
<img src="https://github.com/ruizhecao96/CMGAN/blob/main/Figures/Overview.PNG" width="600px">

<img src="https://github.com/ruizhecao96/CMGAN/blob/main/Figures/Table.PNG" width="600px">

## Long version citation:
```
@misc{abdulatif2022cmgan,
  title={CMGAN: Conformer-Based Metric-GAN for Monaural Speech Enhancement}, 
  author={Abdulatif, Sherif and Cao, Ruizhe and Yang, Bin},
  year={2022},
  eprint={2209.11112},
  archivePrefix={arXiv}
}
```


## Short version citation:
```
@inproceedings{cao22_interspeech,
  author={Cao, Ruizhe and Abdulatif, Sherif and Yang, Bin},
  title={{CMGAN: Conformer-based Metric GAN for Speech Enhancement}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={936--940},
  doi={10.21437/Interspeech.2022-517}
}
```
