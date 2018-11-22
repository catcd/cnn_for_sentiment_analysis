# ``cnn4SA v1.0``
## CNN with various region sizes for Sentiment Analysis

cnn4SA, version ``1.0``, is a project that was developed with 3 main functions:

- Analyze the sentiment of a given paragraph
- Train new models with given corpora that follow the format
- HTTP service for Sentiment Analysis

The model is trained on ``50.000`` IMDB movie reviews.

## Table of Contents
- [cnn4SA v1.0](#--cnn4sa-v10--)
  * [1. Installation](#1-installation)
  * [2. Usage](#2-usage)
    + [Train the models](#train-the-models)
    + [Evaluate the trained models](#evaluate-the-trained-models)
  * [3. HTTP service](#3-http-service)
    + [Start service interface](#start-service-interface)
    + [Service API](#service-api)
    + [Process new input](#process-new-input)
##

## 1. Installation

This program was developed using **Python** version **3.5** and was tested on **Ubuntu 16.04** system. We recommend using Anaconda 3 newest version for installing **Python 3.5** as well as **numpy**, although you can install them by other means. 

Other requirements: 
 1. **numpy**
```sh
# Included in Anaconda package
```

 2. **scipy**
```sh
# Included in Anaconda package
```

 3. **Flask** 
```sh
$ pip install Flask
```

 4. **Tensorflow** 
```sh
$ pip install tensorflow     # Python 3.n; CPU support 
$ pip install tensorflow-gpu # Python 3.n; GPU support 
```
If you are install tensorflow with GPU support, please follow the instructions on the official document to install other required libraries for your platform. Official document can be found at [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/)  

 5. **nltk**
```sh
$ conda install -c anaconda nltk
```
You should download Punkt Tokenizer Models by the command ``nltk.download('punkt')``

 6. **sklearn**
```sh
$ conda install -c anaconda scikit-learn
```

## 2. Usage 
### Train the models
Run python file ``train.py`` to train the model and evaluate on benchmark dataset.
Commands: 
```sh
$ python train.py --help
usage: train.py [-h] -train TRAIN [-val VAL] [-test TEST] [-e E] [-p P] [-b B]
                [-pre PRE] -name NAME

Train Multi-region-size CNN for Sentiment Analysis

optional arguments:
  -h, --help    show this help message and exit
  -train TRAIN  Train data
  -val VAL      Validation data (1vs9 for validation on 10 percents of
                training data)
  -test TEST    Test data
  -e E          Number of epochs
  -p P          Patience of early stop (0 for ignore early stop)
  -b B          Batch size
  -pre PRE      Pre-trained weight
  -name NAME    Saved model name

```

All hyper-parameter is set default to the heuristic values. You can change every setting of the model or try the default one. To change the config, edit the constants in the ``config.py``.

**Example**: train with default setting
```sh
$ python train.py -train data/train_data/train -val 1vs9 -test data/train_data/test -name imdb_sa
```

This command means:
 - Train with the data in ``data/train_data/train``.
 - Use 10 percents of training data fo validation.
 - Train with the data in ``data/train_data/test``.
 - Save the trained weight with the name ``imdb_sa``

### Evaluate the trained models 
After train the model, the model is used to predict on the provided testing dataset. The result is printed out in the end of the output stream.

The result is printed in the format: 
```sh
...
[INFO] Train model... finished in xx.xx s.
[INFO] Test model... started.
Testing result:	P=0.xxx	R=0.xxx	F1=0.xxx
[INFO] Test model... finished in 108.325s.

```

## 3. HTTP service
### Start service interface
Run python file ``service.py`` to start the HTTP service. Commands:

```sh
$ python service.py --help
usage: service.py [-h] -pre PRE

Service Multi-region-size CNN for Sentiment Analysis

optional arguments:
  -h, --help  show this help message and exit
  -pre PRE    Pre-trained weight

```

**Example**: train with default setting
```sh
$ python service.py -pre imdb_sa
```

This command means:
 - Use the trained weight with the name ``imdb_sa``

### Service API
Online service supports HTTP request on same domain. API on v1.0 is:
 - ``hostname/process``: process and return the label for new input.
 
The interface of each API is described in next sections.


### Process new input
**URL**: hostname/process  
**Method**: POST  
**Headers**:  
```content-type: application/json```

**Input data format**: JSON formatted string on Body of the request.

```json
{
  "input": [list of string]
}
```

**Response data format**: result in JSON formatted in Body of response.
```json
{
  "output": [list of [float, float])]
}
```


**Example**:  
_Request_:
```json
{
  "input": [
    "I don't like this boring movie.",
    "For me, this is more: this is the definitive film. 10 stars out of 10."
  ]
}
```

_Response_:  
```json
{
  "output": [
    [
      0.9467412233352661,
      0.05325876176357269
    ], [
      0.3139919936656952,
      0.6860079765319824
    ]
  ]
}
```
