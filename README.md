# PyTorch

Implementations of different ML models in PyTorch, along with training and evaluation scripts. Currently includes implementations of CNN and transformer model architectures.

## Quickstart

Install the necessary python libraries:

```
pip install -r requirements.txt
```

Train and evaluate a simple CNN model on a test dataset:

```
python CNN/main.py
```

## Contents

Within each sub-directory are classes which implement different model architectures. Included so far are:
* ```CNN/```
  * ```CNN.py``` : basic CNN with arbitrary configuration dimensions. Quick entry point which enables fast model training on a CPU.
  * ```VGG16.py``` : implementation of the VGG16 architecture (configuration D) as outlined in https://arxiv.org/pdf/1409.1556
* ```transformer/```
  * ```transformer.py``` : (WIP) implementation of the transformer architecture outlined in https://arxiv.org/pdf/1706.03762

## Usage

NOTE: ```transformer``` is still WIP so some of the below may not yet have been implemented.

Within each model folder is a ```main.py``` script for performing model training and evaluation. At the top of the main function there are configuration variables which can be altered in order to change the training hyperparameters, as well as specifying whether to only perform model training, model evaluation or both.

The model architecture to use can be selected as a command line argument as (e.g. for the CNN entrypoint):

```
python CNN/main.py [--model]
```

where ```--model``` has options ```CNN``` or ```VGG16```, indicating which model file to load the model architecture from.