# Thesis Project for Master of Arts in Computational Linguistics 

## _Acoustic Dialect Recognition Using Deep Learning_ Ryan Callihan

### Seminar für Sprachwissenschaft - Karl Eberhards Universität - Tübingen, Germany

#### Abstract

Dialect Identification (DID) is a well defined task with text data and models using both text and audio data have been shown to work especially well. However, the task of DID when using only audio signals is a bit more complicated and, as such, is much more difficult to achieve good and reliable results. In this thesis, the task of DID using only audio signals was investigated and different methods of feature extraction and machine learning were experimented with. The audio feature extraction methods explored were: Mel-spectrograms, Mel-frequency cepstral coefficients (MFCC) and Delta-Spectral Cepstral Coefficients DSCC. Each method is widely used in the field of speech recognition but their effects on dialect recognition were a bit different. The machine learning techniques used were Recurrent Neural Networks (RNN), Convolutional Neural Networks (CNN), and a hybrid combination of the two. Two corpora were used: one for Arabic, and one for English. The methods listed above were tried on both corpora and the results were analysed. 

#### Corpora 

##### MGB-3 Arabic Dialect Corpus

MGB-3 Corpus WAV files located (HERE)[http://alt.qcri.org/resources/ArabicDialectIDCorpus/varDial2017_Arabic_Dialect_Identification/train/]

##### FRED-s English Dialect Corpus

FRED-s corpus WAV files located (HERE)[https://fred.ub.uni-freiburg.de/]

#### Delta-Spectral Cepstral Coefficient Python Implementation
Code can be found (HERE)[https://github.com/ryancallihan/thesis_project/blob/master/utils/audio_utils.py]

#### Results

| Model   | MGB3-Accuracy | MGB3-Loss | FRED-Accuracy | FRED-Loss |
|---------|---------------|-----------|---------------|-----------|
| RNN     | 0.65          | 1.05      | 0.71          | 1.21      |
| CNN     | 0.66          | 1.60      | 0.70          | 1.55      |
| CNN+RNN | 0.82          | 0.66      | 0.81          | 0.81      |

#### Conclusion

In this study, different methods for the task of dialect identification were tested. First, 3 different neural network architectures were used: a Recurrent Neural Network, a Convolutional Network, and a hybrid between the two. Also, 4 different audio processing techniques were used: MFCC, Mel-Spectrograms, MFCC + DSCC, and Mel Spec + DSCC. Two datasets and languages were used as well: the MGB-3 corpus of Arabic dialects and the FRED-S corpus of English dialects. 
	
A hybrid CNN+RNN model with Mel-Spectrogram features was shown to work the best for the task. A simple RNN was also a simple and far less computationally intensive method as well. A CNN alone was more prone to overfitting.
	
Adding DSCC features to the model had a negative effect on classification, which is not true for speech recognition.
	
The study ended with a qualitative comparison of the dialects using misclassification as a metric.
