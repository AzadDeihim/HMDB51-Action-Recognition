# HMDB51-Action-Recognition
Created several models to classify actions in HMDB51 dataset.  Used models consisting of convolutional and pooling layers for feature extraction followed by recurrent neural networks for classification.  Devised novel two-stream architecture capturing spatial and temporal features of input videos, in conjunction with Mask R-CNN for background removal, with bidirectional long-short term memory for classification, earning 80% accuracy on the test set. 

Compatible with HMDB51 Dataset. 
https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

Each model is named 'model name'.ipynb

Before running each model, convert the dataset to tensors by running VideosToTensors.ipynb.
If you want to run MaskRCNN-LSTM.ipynb, the dataset must first be preprocessed by running VideosToMask.ipynb.
