# QTZ4TSC
**On the Post-Training Quantization of Deep Convolutional Neural Network for Time Series Classification**, AALTD 2026.
Source code and experimental results.

[11th Workshop on Advanced Analytics and Learning on Temporal Data (AALTD 2026)](https://ecml-aaltd.github.io/aaltd2026/program.html)


## Usage

`eval_*.py --model {inception} --dataset {tsc_ucr,mtsc_uea,monster} --expid EXPID --quantization QUANTIZATION`


### Requirements

* `tensorflow==2.16.2`
* `numpy==1.26.4`
* `aeon==1.4.0`
