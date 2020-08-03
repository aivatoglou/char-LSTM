import random
import torch

# Turn string into list of numbers
def char_to_tensor(string, all_characters):
    
    tensor = torch.zeros(len(string)).long()
    for i in range(len(string)):
        tensor[i] = all_characters.index(string[i])
    return tensor

# Return a random training chunk
def random_chunk(chunk_len, batch_size, mode, data, data_len, all_characters):

    input_chunk = torch.LongTensor(batch_size, chunk_len)
    target_chunk = torch.LongTensor(batch_size, chunk_len)
    for i in range(batch_size):
        start_index = random.randint(0, data_len - chunk_len -1)
        end_index = start_index + chunk_len + 1
        chunk = data[start_index:end_index]
        input_chunk[i] = char_to_tensor(chunk[:-1], all_characters)
        target_chunk[i] = char_to_tensor(chunk[1:], all_characters) 
    if torch.cuda.is_available():
        input_chunk = input_chunk.cuda()
        target_chunk = target_chunk.cuda()
    return input_chunk, target_chunk
