import torch

# Train the net
def net_train(input_seq, target_seq, chunk_len, decoder, batch_size, decoder_criterion, decoder_optimizer):
    
    hidden = decoder.init_hidden(batch_size)

    decoder.train()
    loss = 0
    if torch.cuda.is_available():
        hidden = (hidden[0].cuda(), hidden[1].cuda())
    for i in range(chunk_len):
        output, hidden = decoder(input_seq[:,i], hidden)
        loss += decoder_criterion(output.view(batch_size, -1), target_seq[:,i])
    
    decoder.zero_grad()
    loss.backward()
    decoder_optimizer.step()
    return loss.item() / chunk_len
