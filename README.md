In recent years, deep learning approaches have obtained very high performance across many Natural Language Processing
tasks like Sentiment Analysis, Named Entity Recognition, Text Classification, Document Classification, Topic Modeling, and
Web search. Most of these tasks in NLP are sequence modeling i.e. mapping a fixed-length input with a fixed-length output
where the length of the input and output may differ.
The traditional machine learning models and neural networks cannot capture the sequential information present in the text.
Therefore, people started using recurrent neural networks (RNN and LSTM) because these architectures can model
sequential information present in the text. However, these recurrent neural networks have their own set of problems. One
major issue is that RNNs cannot be parallelized because they take one input at a time. In the case of a text sequence, an RNN
or LSTM would take one token at a time as input. So, it will pass through the sequence token by token. Hence, training such a
model on a big dataset will take a lot of time.
Another problem in Natural Language Processing is the shortage of training data as this field is diverse, and the most taskspecific
dataset contains only a few thousand or a few hundred training examples. To address these problems researchers
have developed a variety of techniques for training general-purpose language representation models using an enormous
amount of unannotated text on the web. This pre-trained model can then be fine-tuned on small data NLP tasks which will
result in substantial accuracy improvements compared to training on these datasets from scratch.
One such Pretrained model is BERT stands for Bidirectional Encoder Representations from Transformers released in late
2018 by Google. BERT is designed to pre-train deep bidirectional representations from an unlabeled text by jointly
conditioning on both left and right context in all layers
So my Project Proposal is to utilize the pre-trained BERT model by fine-tuning the parameters and train the new model for
our performing Named Entity Recognition.

Approach:
With minimal architectural modification, I am planning to fine tune BERT and trained the model on MIT Movie Corpora dataset for
performing Named Entity Recognition.
Current State-of-the-art
Researchers in the NLP has been developing various domain-specific language representation model pre-trained on
large-scale biomedical corpora for performing various application in the NLP field. Some of the models are BioBERT
and Roberto.

Dataset:
The Public Dataset which I will be using will be MIT Movie Corpora. MIT Movie Corpora is an openly available dataset developed by
the MIT Lab for Computational Physiology, comprising of 24 labels of unstructured text sentences.
