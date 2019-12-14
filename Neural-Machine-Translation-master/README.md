# Neural-Machine-Translation
Implementing NMT on an IWSLT dataset, using sequence-to-sequence (seq2seq) models, in Tensorflow

# Dependencies
* Python(used v2.7.12)
* tensorflow(used v1.4.1)
* nltk(used v3.3)

# Usage
Code to be run in Linux terminal to train and evaluate the model : 

`$ python main_nmt.py`

# Code Organization
* ```main_nmt.py``` : This file contains the main program. This directs the model to run training/evaluation as directed.
* ```data_preparation.py``` : This file contains code to extract data from the datasets used(refer below) and preprocessing
* ```parameters.py``` : This file defines a function initialising all the parameters and hyperparamers used.
* ```model_attention.py``` : This file defines a class, AttentionModel, that defines the architecture of the encoder-decoder NMT model.
* ```train_nmt.py``` : This file creates train and test/validation instances. It trains and evaluates the model. Checkpoints are created periodically, with the latest five checkpoints stored at any time in the specified output directory. The model is loaded from the latest checkpoint present, or if none are available, a new model is created.
* ```basic_functions.py``` : This file defines functions for different parts of the neural network model architecture and for evaluation.
* ```additional_functions.py``` : The file defines functions to load data and and to format the output.
* ```calculate\_bleu\_score.py``` : This file calculates the BLEU-4 score given two files to be compared. Usage : `python calculate_bleu_score.py /path/to/reference_file /path/to/predicted_file`

# Datasets Used
English-Vietnamese parallel corpus of TED Talks, provided by the [IWSLT Evaluation Campaign](https://sites.google.com/site/iwsltevaluation2015/).
Preprocessed data from [The Stanford NLP group](https://nlp.stanford.edu/projects/nmt/) was used to train and test the models.
* ```Datasets/train.en``` (train source set)
* ```Datasets/train.vi``` (train target set)
* ```Datasets/tst2013.en``` (validation source set)
* ```Datasets/tst2013.vi``` (validation target set)
* ```Datasets/tst2012.en``` (test source set)
* ```Datasets/tst2012.vi``` (test target set)
* ```Vocabulary_Files/vocab.en``` (source vocabulary)
* ```Vocabulary_Files/vocab.vi``` (target vocabulary)

# References
* [Neural Machine Translation(seq2seq) Tutorial](https://www.tensorflow.org/tutorials/seq2seq) by Tensorflow and their [source code](https://github.com/tensorflow/nmt) 