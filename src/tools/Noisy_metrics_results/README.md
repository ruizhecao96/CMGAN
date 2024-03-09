# Compute Metrics Validation

## Overview
This repository contains log files for validating the `compute_metrics.py` Python tool against the original MATLAB source code provided by Prof. Philipos C. Loizou in his seminal book “Speech Enhancement Theory and Practice.”

## Dataset
The dataset used for validation consists of 824 noisy tracks from the Voice Bank+DEMAND dataset, resampled at 16 kHz. The dataset is available for download at the following link:
[Noisy Tracks Dataset](https://drive.google.com/file/d/1TGoOYP5YLeCB3lTAACavMSYhuuIdTtGw/view?usp=sharing)

## Validation Process
Our validation process involved comparing results from the original MATLAB script with our Python version, which is available in this GitHub repository. The comparison was made using the provided dataset. 

Both the MATLAB and Python versions reported identical results, as presented in our log files.

## Original MATLAB Source Code
The original MATLAB source code by Prof. Philipos C. Loizou can be accessed through the following link:
[Original MATLAB Code](https://www.crcpress.com/downloads/K14513/K14513_CD_Files.zip)
