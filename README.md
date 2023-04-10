# LabelSemantics
> Code for Label Semantics for Few Shot Named Entity Recognition

## Requirements
1. First download the required dataset from [here](https://cemantix.org/conll/2012/data.html).
2. Download training, develpment and test data for CoNLL 2012 and place it in the ../NewDataset folder.
3. Download the OntoNotes Release 5.0 from [here](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/KPKFPI) and place it in the ../NewDataset folder.
4. Use the scripts present in this repo to generate the required files for training and testing.

## Training
1. Before training change the `train_path`, `test_path`, and `dev_path` present in `LabelSemantics.py` to the path of the training data.