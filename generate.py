from helpers import char_to_tensor
import torch

# Generate a sequence of characters
def char_generate(decoder, all_characters, prime_seq, predict_len, temperature):

    hidden = decoder.init_hidden(1)
    prime_seq_torch = char_to_tensor(prime_seq,all_characters).unsqueeze(0)

    if torch.cuda.is_available():
        hidden = (hidden[0].cuda(), hidden[1].cuda())
        prime_seq_torch = prime_seq_torch.cuda()
    
    predicted = prime_seq
    for i in range(len(prime_seq) - 1):
        output, hidden = decoder(prime_seq_torch[:,i], hidden)

    last_char = prime_seq_torch[:,-1]
    for i in range(predict_len):

        output, hidden = decoder(last_char, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        last_char = char_to_tensor(predicted_char, all_characters).unsqueeze(0)
        if torch.cuda.is_available():
            last_char = last_char.cuda()

    return predicted
    