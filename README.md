<h1>Fine-Tuning BERT for Named Entity Recognition task using PyTorch</h1>

<p>
  In this project we use BERT with huggingface PyTorch library to quickly and efficiently fine-tune a model to get near state of the art performance in Named Entity Recognition.</p>
  <p>
  NER is widely used NLP task tht tries to locate the names and entities contained in a piece of text, commonly things like people organisation, locations etc. 
  
  </p>
  
  <h2>Dataset</h2>
  
  <p>
  
  Dataset used in this model is <strong>MIT Movie Corpus</strong>.  The MIT Movie Corpus is a semantically tagged training and test corpus in BIO format. The eng corpus are simple queries, and the trivia10k13 corpus are more complex queries.
  
  </p>
  https://groups.csail.mit.edu/sls/downloads/movie/
  
  ![image](https://user-images.githubusercontent.com/57468338/121114664-168fef00-c7e2-11eb-9908-7b838f1ff5a8.png)

  
  <h2>NER Tags and IOB Format  </h2>
  
  <p>
  Both the train and test datasets are single files containing movie-related text where each word has a NER tag specifying it as one of the following entities.
The NER tag follows a special format used widely in NER literature called IOB format(Inside,Outside and Beginning Format).
  <ul>
    <li>O:This Tag means the word is not part of the entity  </li>
    <li>B:this tag means the word is either a single word entity name or else the first word in a multi-word entity name</li>
    <li>I: this tag means the word is part of a ulti-word entity but is not the first word in the full entity name  </li>
</p>
  
  ![image](https://user-images.githubusercontent.com/57468338/121115109-b9e10400-c7e2-11eb-9fb1-9b4f4784a28b.png)


<h2>Data Preparation:   </h2>

<p>
  First of all we process our data into desired format for fine tuning. We separate the data into sentencs and label.
  <ul>
    <li>Sentences containing a list of tokenized sentences   </li>
    <li>Lable containing a list of corresponding IOB   </li>
</ul>
Along with this we also have a label map dictionary that will map our tags to integres during the fine tuning process

</p>

<h2>Tokenization and Input formatting </h2>
<p>
  We need to map the words to theirs IDs in the BERT vocabulary and forwords that arent in the BERT vocabulary, the tokenizer will break these down into subwords. BERT requires thatwe prepend the special  <strong>[CLS]  </strong> token to the beginning of every sentence and append the <strong> [SEP]</strong> token at the end. To support the batch processing of the inputs all the sentences should be padded out(with special [PAD] token) to the same length.
  
  </p>
  
  
 <p>
  We measure the length of all the sentences and look at the distribution of lengths in order to choose a sensible "max_len" value for padding our sentences to.
  
  </p>
  
![image](https://user-images.githubusercontent.com/57468338/121116938-54424700-c7e5-11eb-92bb-9908a89b7f9f.png)
  
 <h2> Training and Validation Split </h2> 
 <p>
 We Divided our training set to use 90% for training and 10% for validation. For  we used TenSorDataset for combining training inputs into tensors.
 We'll also create an iterator for our dataset using the torch DataLoader class. This helps save on memory during training because, unlike a for loop, with an iterator the entire dataset does not need to be loaded into memory.
 </p>
 
 
 
 
  
  </p>
 
