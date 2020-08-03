import torch

# Evaluate the net from the validation data
def net_evaluate(input_seq, target_seq, chunk_len, decoder, decoder_criterion, batch_size):

    hidden = decoder.init_hidden(batch_size)

    decoder.eval()
    val_loss = 0
    if torch.cuda.is_available():
        hidden = (hidden[0].cuda(), hidden[1].cuda())
    for i in range(chunk_len):
        output, hidden = decoder(input_seq[:,i], hidden)
        val_loss += decoder_criterion(output.view(batch_size, -1), target_seq[:,i])
    return val_loss.item() / chunk_len
