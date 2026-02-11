---
library_name: transformers
license: cc-by-sa-4.0
base_model: jcblaise/roberta-tagalog-base
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
model-index:
- name: gaslighting-detector-tactic-roberta-tagalog-base
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# gaslighting-detector-tactic-roberta-tagalog-base

This model is a fine-tuned version of [jcblaise/roberta-tagalog-base](https://huggingface.co/jcblaise/roberta-tagalog-base) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3560
- Accuracy: 0.9111
- F1: 0.9124

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|
| 1.1375        | 1.0   | 80   | 0.6429          | 0.7729   | 0.7566 |
| 0.5010        | 2.0   | 160  | 0.4991          | 0.8352   | 0.8321 |
| 0.2636        | 3.0   | 240  | 0.5214          | 0.8315   | 0.8248 |
| 0.1017        | 4.0   | 320  | 0.5247          | 0.8443   | 0.8417 |
| 0.0487        | 5.0   | 400  | 0.5422          | 0.8443   | 0.8422 |


### Framework versions

- Transformers 5.0.0
- Pytorch 2.7.1+cu118
- Datasets 4.5.0
- Tokenizers 0.22.2
