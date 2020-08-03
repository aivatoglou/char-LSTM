import numpy as np
import torch
import json
import time
import re

from model import RNN
from train import net_train
from helpers import random_chunk
from evaluate import net_evaluate
from generate import char_generate

# Vocabulary reduction
def load_data(path):

    f = open(path, 'r')
    text = f.read()
    
    text = re.sub("\'", "", text) # remove backslash-apostrophe
    text = re.sub("[^a-zA-Z]", " ", text) # remove non alphabet
    text = ' '.join(text.split()) # remove whitespaces
    text = text.lower() # lowercase words 
    return text

hyperparameters_file = open('hyperparameters.json')
hyperparameters = json.load(hyperparameters_file)

# Hyperparameters
hidden_size = hyperparameters['train']['hidden_size']
batch_size = hyperparameters['train']['batch_size']
chunk_len = hyperparameters['train']['chunk_len']
n_epochs = hyperparameters['train']['n_epochs']
n_layers = hyperparameters['train']['n_layers']
lr = hyperparameters['train']['lr']

# Load train and validation data
train_data = load_data('data/wiki.train.txt')
validation_data = load_data('data/wiki.valid.txt')

# Print train-dataset statistics
text_len = len(train_data)
all_characters = tuple(sorted(set(train_data)))
n_characters = len(all_characters)
print('Total characters: {} - Total vocab: {}'.format(text_len, n_characters))

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
decoder_criterion = torch.nn.CrossEntropyLoss()

if not torch.cuda.is_available():
    print('CUDA is not available.  Training on CPU ...\n')
else:
    print('CUDA is available!  Training on GPU ...\n')
    decoder.cuda()   

start = time.time()
for epoch in range(1, n_epochs):
    
    # Get a random training chunk for each epoch
    input_chunk, target_chunk = random_chunk(chunk_len, batch_size, 'train', train_data, len(train_data), all_characters)
    # Train and calculate the loss
    loss = net_train(input_chunk, target_chunk, chunk_len, decoder, batch_size, decoder_criterion, decoder_optimizer)       
    # Print every 100 epochs and calculate the validation loss
    if epoch % 100 == 0:
        input_valid_chunk, target_valid_chunk = random_chunk(chunk_len, batch_size, 'valid', validation_data, len(validation_data), all_characters)
        print('[%.4f (%d %d%%) Train loss: %.4f Valid loss: %.4f Perplexity: %.2f]' \
              % (time.time()-start, epoch, epoch / n_epochs * 100, loss, \
                 net_evaluate(input_valid_chunk, target_valid_chunk, chunk_len, decoder, decoder_criterion, batch_size), np.exp(loss)))

# Generate
prefix = hyperparameters['generate']['prefix']
temperature = hyperparameters['generate']['temperature']
print(char_generate(decoder, all_characters, prefix, chunk_len, temperature))
