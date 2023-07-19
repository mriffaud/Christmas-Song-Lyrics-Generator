# Creating a Christmas Song Lyrics Generator with Machine Learning

![banner](https://media.istockphoto.com/id/1135217654/photo/image-of-christmas-turntable-vinyl-record-player-sound-technology-for-dj-to-mix-play-music.jpg?s=612x612&w=0&k=20&c=F8vKCkvsLwi5ynGimyG4QkX9KdSDkswUUUlqPqwY8IQ=)

In the spirit of the holiday season, we are going to embark on a project to create a machine-learning model that generates Christmas song lyrics. By training the model on a dataset of Christmas songs and carols, we can generate new and unique lyrics that capture the festive spirit. 

In this article, we will walk through the process of cleaning the data, training the model, and generating lyrics. We will be using PyTorch, a popular machine-learning framework, for the implementation. So, let's dive in and spread the holiday cheer with our own AI-generated Christmas songs!

---
## Table of Content
- [1. Data ](#data)
- [2. Importing Library ](#step1)
- [3. Data Preprocessing ](#step2)
  * [3.1 Downloading and Reading the Dataset ](#step2.1)
  * [3.2 Cleaning the Lyrics ](#step2.1)
  * [3.3 Tokenizing the Lyrics ](#step2.3)
- [4. Model Training ](#step3)
  * [4.1 Defining the Model Architecture ](#step3.1)
  * [4.2 Training Loop ](#step3.2)
- [5 Lyrics Generation ](#step4)
- [6 Conclusion ](#step5)


---
<a name="data"></a>
## Data
* **Data Format**: The data is stored in a CSV (Comma Separated Values) file format. Each row in the CSV file represents a single line of a Christmas song or carol.

* **Data Source**: The data is obtained from a GitHub Gist. A Gist is a simple way to share code snippets, text, and other types of content on GitHub. The URL to the Gist containing the input data is as follows:

   URL: https://gist.githubusercontent.com/DeastinY/899d532069febdb969d50eb68b7be583/raw/d4c2b7d6cd58639274fa2f061db6905c58853947/input.txt

* **Data Contents**: Each row in the CSV file contains a line of text, which represents a line from a Christmas song or carol. The data may include song titles, verses, refrains, choruses, bridges, and other elements commonly found in songs.

---
<a name="step1"></a>
## Importing Library
The code below imports the necessary libraries and modules for data preprocessing, model training, and lyrics generation using PyTorch.

```python
import re
import random
import numpy as np
import pandas as pd
import string, os 
import requests
import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
```
Here's a breakdown of each library imported in the code:
* `re`: This library provides support for regular expressions, which are used for pattern matching and text manipulation.
* `random`: This library allows generating random numbers, selecting random elements from a list, and shuffling sequences randomly.
* `numpy` (np): This library is used for numerical operations in Python, providing support for arrays, matrices, and mathematical functions.
* `pandas` (pd): This library is used for data manipulation and analysis. It provides data structures like dataframes that facilitate working with tabular data.
* `string`: This library provides a collection of string constants and helper functions for string operations, such as character manipulation and formatting.
* `os`: This library provides a way to interact with the operating system. It allows accessing and manipulating files and directories, among other system-related tasks.
* `requests`: This library allows sending HTTP requests to retrieve data from URLs. It is used to download the CSV file from a given URL in the code.
* `io`: This library provides tools for handling I/O operations, including reading and writing data to streams and files. In this code, it is used to read the content of the downloaded file.
* `torch`: This library is PyTorch, a popular deep learning framework. It provides functionality for creating and training neural networks, as well as tensor operations and GPU acceleration.
* `torch.nn` (nn): This module provides classes and functions for defining and training neural networks in PyTorch. It includes various layers, loss functions, and optimization algorithms.
* `torch.utils.data`: This module provides tools for creating and working with datasets and data loaders in PyTorch. It allows efficient loading and batching of data during model training.
* `torch.nn.utils.rnn`: This module provides utility functions for working with recurrent neural networks (RNNs) in PyTorch. It includes functions for padding sequences and packing padded sequences.

The last two lines suppress any warning messages that might be displayed during the execution of the code.

---
<a name="step2"></a>
## Data Preprocessing
Before we can start training our machine learning model, we need to preprocess the data. In this section, we will download a dataset of Christmas song lyrics and clean it to remove unnecessary elements.


<a name="step2.1"></a>
### Downloading and Reading the Dataset
We will begin by downloading a dataset of Christmas song lyrics from a publicly available source. Using the `requests` library, we can fetch the dataset from a URL. Then, using the `pandas` library, we will read the downloaded dataset into a pandas DataFrame.

```python
# Downloading the csv file from your GitHub account
url = "https://gist.githubusercontent.com/DeastinY/899d532069febdb969d50eb68b7be583/raw/d4c2b7d6cd58639274fa2f061db6905c58853947/input.txt"
download = requests.get(url).content

# Reading the text file
df = pd.read_csv(io.StringIO(download.decode('utf-8'))
                 ,delimiter= '/t' #'\s+'
                 ,index_col=False
                 ,header=None
                 ,on_bad_lines='skip')

# Transforming the file into a list
lyrics = df[0].astype(str).tolist()
```

We first define the URL path of the CSV file containing the lyrics and use the `requests.get()` function to download the content of the file. The `content` attribute of the response object is assigned to the variable `download`.

Next, we use the `pd.read_csv()` function from the pandas library to read the downloaded CSV file. The `io.StringIO()` function is used to convert the downloaded content to a readable format:
* `decode('utf-8')` method is applied to decode the content as UTF-8
* `delimiter` parameter is set to `'/t'` to specify that the CSV file is tab-separated
* `index_col` parameter is set to `False` to prevent treating any column as an index
* `header` parameter is set to `None` to indicate that there is no header row in the file
* `on_bad_lines` parameter is set to `'skip'` to skip any problematic lines in the file

We then retrieve the first column (`[0]`) of the DataFrame `df` and converts it into a string data type. The `.tolist()` method is then used to convert the column into a Python list. The resulting list, `lyrics`, contains the lyrics of the songs from the CSV file.


<a name="step2.2"></a>
### Cleaning the Lyrics
Once we have the dataset loaded, we need to clean the lyrics to remove any unnecessary elements that might affect the training process. We will remove special characters and stopwords, and eliminate repeating sentences. By performing these steps, we can create a clean and concise dataset for training our model.

```python
# Define the list of stopwords and special characters
stopwords = ['intro', 'verse', 'refrain', 'chorus', 'bridge', 'version', 'repeat']
special_chars = r'[\[\]\(\)\{\}\.,;:?!\'"]'
```

In the codes above, we define two variables: `stopwords` and `special_chars`. 
* `stopwords` is a list of words that you want to remove from the lyrics because they are commonly used and do not contribute much to the meaning.
* `special_chars` is a regular expression pattern that matches various special characters commonly found in text. These characters will be removed from the lyrics.

```python
# Clean the lyrics
cleaned_lyrics = []
for line in lyrics:
    line = re.sub(special_chars, '', line.lower())  # Remove special characters
    line = ' '.join([word for word in line.split() if word not in stopwords])  # Remove stopwords
    if line not in cleaned_lyrics:  # Remove repeating sentences
        cleaned_lyrics.append(line)
```

Then we clean the lyrics by performing the following steps:
* Initialise an empty list called `cleaned_lyrics` to store the cleaned lyrics.
* Iterate over each line in the `lyrics` list.
* Remove special characters from the line using `re.sub()` function. The `re.sub()` function replaces the matches of the `special_chars` pattern with an empty string, effectively removing them from the line. `line.lower()` is used to convert the line to lowercase before removing the special characters.
* Split the line into individual words using the `split()` function and create a list comprehension to filter out any stopwords from the line.
* Join the filtered words back into a single line using the `' '.join()` function. This ensures that the words are separated by a space.
* Check if the cleaned line is already present in the `cleaned_lyrics` list. This is done to remove any repeating sentences from the lyrics.
* If the cleaned line is not already present, append it to the `cleaned_lyrics` list.

After the execution of this code block, the `cleaned_lyrics` list will contain the lyrics without stopwords, special characters, and repeating sentences.


<a name="step2.3"></a>
### Tokenizing the Lyrics
To prepare the lyrics for training, we need to tokenize them into individual words. This step allows us to represent the lyrics as numerical sequences that can be fed into our model. We will create a vocabulary of unique words and create mappings between the words and their corresponding indices.

```python
# Tokenize the lyrics
tokens = []
for line in cleaned_lyrics:
    tokens.extend(line.split())
```

This code initialises an empty list called `tokens` to store individual words from the cleaned lyrics. It iterates over each line in the `cleaned_lyrics` list and splits each line into words using the `split()` method. The individual words are then added to the `tokens` list using the `extend()` method.

```python
# create a vocabulary list
vocab = list(set(tokens))
```

This code creates a vocabulary list by converting the `tokens` list to a set using the `set()` function. This step removes any duplicate words, ensuring that each word appears only once in the vocabulary. The resulting set is then converted back to a list using the `list()` function.

```python
# Create word-to-index and index-to-word mappings
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
```

These lines create two dictionaries: `word_to_idx` and `idx_to_word`. The `word_to_idx` dictionary maps each unique word in the vocabulary to a unique index, with the word as the key and the index as the value. The `enumerate()` function is used to generate pairs of indices and corresponding words from the `vocab` list.

The `idx_to_word` dictionary reverses the mapping by mapping each index to its corresponding word. Again, the `enumerate()` function is used to generate the pairs of indices and words from the `vocab` list, but in this case, the index is the key and the word is the value.

```python
# Convert the lyrics to numerical sequences
sequences = []
for line in cleaned_lyrics:
    sequence = [word_to_idx[word] for word in line.split()]
    sequences.append(sequence)
```

This code initializes an empty list called `sequences` to store the numerical sequences representing the lyrics. It iterates over each line in the `cleaned_lyrics` list and splits each line into words using the `split()` method. For each word in the line, the corresponding index from the `word_to_idx` dictionary is retrieved, and a list of these indices is created for the line. The resulting list of indices is appended to the `sequences` list.

In this section we have tokenized the lyrics by splitting them into individual words, created mappings between words and their corresponding indices, and converted the lyrics into numerical sequences by replacing each word with its corresponding index.

---
<a name="step3"></a>
## Model Training
Now that we have preprocessed our data, we can move on to training our machine-learning model. In this section, we will define our model architecture, set hyperparameters, and train the model using the preprocessed lyrics dataset.


<a name="step3.1"></a>
### Defining the Model Architecture
We will use a GRU-based (Gated Recurrent Unit) model for generating lyrics. The GRU is a type of recurrent neural network (RNN) that is well-suited for sequential data generation tasks. Our model will consist of an embedding layer, a GRU layer, and a fully connected layer for predicting the next word in the sequence.

```python
# Define a custom dataset class for training
class LyricsDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx])
```

This code defines a custom dataset class `LyricsDataset` that inherits from `torch.utils.data.Dataset`. The `__init__` method initializes the dataset object with a list of sequences. The `__len__` method returns the length of the dataset (number of sequences). The `__getitem__` method returns an individual sequence at the specified index as a `torch.LongTensor`.

```python
# Pad the sequences and create data loader
padded_sequences = pad_sequence([torch.LongTensor(seq) for seq in sequences], batch_first=True)
data_loader = DataLoader(LyricsDataset(padded_sequences), batch_size=32, shuffle=True)
```

These lines pad the sequences with the `pad_sequence` function from `torch.nn.utils.rnn` module. The `pad_sequence` function takes a list of tensors (sequences) as input and pads them with zeros to make them equal length, resulting in a tensor of shape `[max_length, batch_size]`. The `batch_first=True` argument ensures that the first dimension of the tensor is the batch size.

The padded sequences are then passed to the `LyricsDataset` class as input to create a dataset object. The `DataLoader` class from `torch.utils.data` is used to create a data loader that batches the dataset into mini-batches of size 32 (`batch_size=32`) and shuffles the data (`shuffle=True`) for training.

```python
# Define the GRU-based model
class LyricsGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LyricsGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        output = self.fc(output)
        return output
```

This code defines the GRU-based model for lyrics generation. It is implemented as a subclass of `nn.Module`. The `__init__` method initializes the model with the vocabulary size, embedding dimension, and hidden dimension of the GRU layer. It defines an embedding layer (`nn.Embedding`), a GRU layer (`nn.GRU`), and a fully connected layer (`nn.Linear`) for the final output. The `forward` method defines the forward pass of the model, where the input `x` is first embedded, then passed through the GRU layer, and finally through the fully connected layer to generate the output.


<a name="step3.2"></a>
### Training Loop
In this subsection, we will define the training loop for our model. We will iterate over the preprocessed lyrics dataset in batches and perform forward and backward passes to optimize the model parameters. We will use the Adam optimiser and the cross-entropy loss function to train our model.

```python
# Set the hyperparameters
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 256
num_epochs = 50
learning_rate = 0.001
```

These lines define the hyperparameters for the model training process. `vocab_size` is the size of the vocabulary, `embedding_dim` is the dimensionality of the word embeddings, `hidden_dim` is the number of hidden units in the GRU layer, `num_epochs` is the number of training epochs, and `learning_rate` is the learning rate for the optimizer.

```python
# Initialize the model
model = LyricsGenerator(vocab_size, embedding_dim, hidden_dim)
```

This line initializes an instance of the `LyricsGenerator` model with the specified hyperparameters.

```python
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

These lines define the loss function and optimizer for training the model. `nn.CrossEntropyLoss()` is used as the loss function, which combines a softmax activation function with the cross-entropy loss. `torch.optim.Adam()` is used as the optimizer, which applies the Adam optimization algorithm to update the model parameters based on the computed gradients.

```python
# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.transpose(1, 2), targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
```

These lines define the training loop that iterates over the specified number of epochs. Within each epoch, the model is trained on batches of data from the `data_loader`. 

- `inputs` are the input sequences for the model, obtained by removing the last word from each batch element.
- `targets` are the target sequences for the model, obtained by removing the first word from each batch element.

The forward pass is performed by passing the `inputs` to the model, which generates predicted outputs. The loss is calculated by comparing the predicted outputs with the `targets` using the `criterion` (cross-entropy loss). 

The optimizer's gradients are set to zero using `optimizer.zero_grad()`, the loss gradients are computed using `loss.backward()`, and the optimizer's `step()` function is called to update the model parameters based on the gradients.

After each epoch, the current loss is printed to track the progress of the training process.

---
<a name="step4"></a>
## Lyrics Generation
After training our model, we can use it to generate new and unique Christmas song lyrics. In this section, we will demonstrate how to generate lyrics using the trained model.

We will randomly select a starting sequence from the preprocessed lyrics dataset and iteratively predict the next word based on the previous predictions. To introduce randomness and diversity into the generated lyrics, we will use temperature-based sampling. The temperature parameter allows us to control the level of randomness in the word selection process.

Finally, we will print the generated lyrics to the console, giving us our very own Christmas song lyrics generated by our machine learning model.

```python
# Generate lyrics using the trained model
start_seq = random.choice(sequences)
input_seq = torch.LongTensor(start_seq).unsqueeze(0)
with torch.no_grad():
    for _ in range(10):  # Generate 10 lines of lyrics
        output = model(input_seq)
        last_word_probs = output[:, -1, :].squeeze().softmax(dim=0)
        
        # Temperature-based sampling
        temperature = 0.8  # Adjust this value for more or less randomness
        scaled_probs = last_word_probs.div(temperature).exp()
        predicted_idx = torch.multinomial(scaled_probs, num_samples=1).item()
        
        # Update the input sequence with the predicted word
        input_seq = torch.cat((input_seq, torch.LongTensor([[predicted_idx]])), 1)
        
        predicted_word = idx_to_word[predicted_idx]
        print(predicted_word, end=' ')
```

```python
silence threefold shiny foot another knew last-minute slice spell crocodile
```

The lines `start_seq = random.choice(sequences)` and `input_seq = torch.LongTensor(start_seq).unsqueeze(0)` are responsible for selecting a starting sequence and converting it into a tensor format that can be fed into the model for lyrics generation:
* `start_seq = random.choice(sequences)` randomly selects a sequence from the list of available sequences (`sequences`). This sequence will serve as the initial input to the lyrics generation process.
* `input_seq = torch.LongTensor(start_seq).unsqueeze(0)` converts the selected starting sequence (`start_seq`) into a PyTorch tensor. The `torch.LongTensor()` function converts the sequence into a tensor of long integers, which is the expected data type for most natural language processing tasks. The `unsqueeze(0)` function adds an extra dimension to the tensor to represent the batch size. This is necessary because models usually expect input in batch format, even when working with a single input sequence.

Next, we use `with torch.no_grad()` context manager is used to temporarily disable gradient calculations during the loop. This is beneficial for inference and generating lyrics because it allows us to perform forward passes through the model without storing gradients or updating model parameters. 

During training, gradients are computed and used to optimise the model parameters, but during inference or generating lyrics, we are only interested in making predictions based on the learned parameters. Disabling gradient calculation reduces memory usage and speeds up the computations, as the computational graph does not need to be stored and the backward pass for gradient computation is not performed. The loop with `range(10)` generates 10 lines of lyrics. Within each iteration, the model predicts the next word based on the input sequence, using the `input_seq` tensor. By disabling gradient calculation and performing only forward passes, we can efficiently generate lyrics without the need for backward passes or gradient updates. So, by using `torch.no_grad()`, we ensure that the generated lyrics are based solely on the learned parameters of the model without any gradient-related computations.

Fowllowing this, we set `output = model(input_seq)` where `model(input_seq)` represents the forward pass of the `model` object. `model` is an instance of the `LyricsGenerator` class, which is a PyTorch module representing the lyrics generation model. When `model(input_seq)` is called, it passes the `input_seq` tensor through the model's layers to obtain the predicted output. More specifically, the `input_seq` tensor contains a batch of input sequences, where each sequence represents a line of lyrics encoded as a sequence of word indices. The tensor is passed through the model's layers, which include an embedding layer, a GRU layer, and a linear layer. These layers collectively process the input and generate an output tensor. The output tensor represents the model's predictions for the next word in each input sequence. Each element of the output tensor corresponds to a word in the vocabulary, and the values in the tensor indicate the model's confidence or probability for each word being the next word in the sequence. By assigning `output = model(input_seq)`, we capture the predicted output tensor for further processing, such as sampling the next word or calculating the loss during training.

Next, we have `last_word_probs` which contains the probabilities of the predicted words being the next word in the sequence. Each value in `last_word_probs` represents the probability of the corresponding word in the vocabulary being the next word, following the context provided by the input sequence. For this, we set:
* `output[:, -1, :]`: This indexing operation selects the last word's predictions from the `output` tensor. The `:` notation in the first dimension indicates that all samples in the batch are selected, and `-1` in the second dimension selects the last time step in the sequence. The `:` notation in the third dimension indicates that all elements in the output vector for the last time step are selected.
* `squeeze()`: This function removes any dimensions with size 1 from the tensor. In this case, it squeezes the tensor to remove the batch dimension, resulting in a 1D tensor.
* `softmax(dim=0)`: This function applies the softmax operation along the 0th dimension of the tensor. Softmax converts the values in the tensor into probabilities, ensuring that they sum up to 1. It transforms the tensor into a probability distribution over the vocabulary, where each value represents the likelihood of a word being the next word in the sequence.

This probability distribution is then used in the subsequent steps of temperature-based sampling to select the next word for generating lyrics where the `temperature` parameter controls the level of randomness in the word selection process during lyrics generation:
* `temperature` is set to a specific value (0.8 in this case), but you can adjust it to a higher or lower value based on your desired level of randomness. A higher temperature, such as 1.0, will introduce more randomness, while a lower temperature, such as 0.5, will make the predictions more focused and deterministic.
* `scaled_probs` is calculated by dividing the probabilities of the last predicted word by the `temperature` value and then applying the exponential function (`exp()`) to rescale the values. Dividing by the temperature increases or decreases the relative differences between the probabilities, resulting in a broader or narrower probability distribution.
* `torch.multinomial()` is used to perform multinomial sampling based on the `scaled_probs`. It randomly selects a word index from the probability distribution, considering the adjusted probabilities. The `num_samples=1` argument indicates that only one sample (word index) should be drawn.
* `predicted_idx` stores the selected word index, which will be used to retrieve the corresponding word from the vocabulary.

By adjusting the `temperature`, you can control the trade-off between randomness and coherence in the generated lyrics. Higher values will result in more diverse but potentially less coherent lyrics, while lower values will make the predictions more focused and coherent but less varied.

We can then update the input sequence with the predicted word and printing the predicted word. For this we first use `torch.cat()` function to concatenate the original `input_seq` tensor with a new tensor representing the predicted word index `predicted_idx`. The `torch.LongTensor([[predicted_idx]])` creates a tensor with the shape `[1, 1]` containing the predicted word index. The `1` as the second argument in `torch.cat()` indicates that the concatenation should be done along the second dimension (columns). This effectively extends the input sequence with the predicted word index, allowing it to be used as input for the next iteration.

We use `predicted_idx` to look up the corresponding predicted word from the `idx_to_word` mapping. `idx_to_word` is a dictionary that maps the index of a word to the word itself. By indexing `idx_to_word` with `predicted_idx`, we obtain the actual word represented by the predicted index.

Finally, we print the `predicted_word` to the console. The `end=' '` argument is used to specify that the print statement should end with a space instead of a newline character. This ensures that the next predicted word will be printed on the same line, creating a continuous line of lyrics.

By updating the `input_seq` tensor and printing the predicted word, the code allows for generating lyrics by iteratively predicting the next word in the sequence.

---
<a name="step5"></a>
## Conclusion
In this article, we have explored the process of creating a Christmas song lyrics generator using machine learning. We learned how to preprocess the data, train the model, and generate lyrics. By leveraging PyTorch, we were able to create a GRU-based model that generates unique lyrics. This project demonstrates the power of machine learning in creative tasks and provides a fun way to celebrate the holiday season. So go ahead, create your own Christmas song lyrics generator and spread the festive cheer!

