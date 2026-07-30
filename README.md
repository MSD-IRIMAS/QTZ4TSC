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


## Cite this work

If you use this work please cite the following:
```bibtex
@inproceedings{Meyer-QTZ4TSC,
  author = {Meyer, Cyril and Ismail-Fawaz, Ali and Forestier, Germain},
  title = {On the Post-Training Quantization of Deep Convolutional Neural Network for Time Series Classification},
  booktitle = {ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data},
  city = {Naples},
  country = {Italy},
  year = {2026}
}
```

## Acknowledgments

This work was supported by a welcome contract funding from the Université de Haute-Alsace. This work was supported by the ANR Terminus project (ANR-25-CE23-2879) of the French Agence Nationale de la Recherche. We acknowledge the High Performance Computing Center of the University of Strasbourg for supporting this work by providing scientific support and access to computing resources. Part of the computing resources were funded by the Equipex Equip@Meso project (Programme Investissements d’Avenir) and the CPER Alsacalcul/Big Data.
