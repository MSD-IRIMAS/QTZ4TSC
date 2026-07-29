# Results

### Filename explained

* BASELINE_model_dataset
  * Model evaluated without modification (baseline) and `quantizer.identity_quantize`
* EXP_inception_monster_opt
  * Results from training script, only use them for `train_time_s`
* QTZ_model_dataset_quantization
  * Model evaluated with quantization


Warning `EXP_inception_UCR.csv` and `EXP_inception_UEA.csv` are from the training and do not use model `_best` but `_last`.

Warning, accuracy on some models differs when comparing baseline, identity, QintZ32 and Qint32.
Differences for UCR and UEA are shown in the following tables, the errors are negligible.


```
         acc1      acc2      acc3      acc4
1    0.836317  0.833760  0.836317  0.836317
35   0.804000  0.804000  0.804000  0.802000
121  0.751256  0.751256  0.750977  0.751256
127  0.893667  0.894000  0.893667  0.893667
```

```
        acc1      acc2      acc3      acc4
9   0.671964  0.672247  0.672247  0.672247
15  0.609489  0.609084  0.609489  0.609489
20  0.294363  0.294065  0.294661  0.294363
```
