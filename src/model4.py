import numpy as np
import pandas
import random
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from midi_parser import Midi, Note

# HYPERPARAMETERS
INPUT_SIZE = 128
HIDDEN_SIZE = 512
OUTPUT_SIZE = 128

class midiRNN4(nn.Module):
    def __init__(self, device):
        super(midiRNN4, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.output_size = OUTPUT_SIZE
        self.num_layers = 3
        self.device = device
        # self.fc0 = nn.Linear(input_size, hidden_size)
        self.emb = nn.Embedding(self.input_size, self.hidden_size, device=device)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout()
        # self.fc2 = nn.Linear(hidden_size,output_size ) 
    
    def reset_hidden(self):
        #TODO remove variable if this works
        self.hidden = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).to(self.device),
                        Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).to(self.device))
 
    def forward(self, X):
        # print("X shape before fc0:", X.shape)
        # out = self.fc0(X.float())
        
        self.emb.to(self.device)
        embeds = self.emb(X.view(1, -1))
        embeds.to(self.device)

        out, self.hidden = self.rnn(embeds.view(1,1,-1), self.hidden) # (block_size, hidden_size)

        out = self.dropout(out)
        # print("OUT shape:", out.shape)
        # features = torch.cat([torch.amax(h, dim=1),
        #                       torch.mean(h, dim=1)], axis=-1)
        out = self.fc1(out.view(1,-1)) # Gets the last timestep
        
        # print("Out prob dist: ", out)
        # print("Highest prob: ", torch.max(out))

        # print("FORWARD Z shape:",z.shape)
        # y = self.activation(z) #(hidden_size, output_size)
        # print("Y shape:",y.shape)
        return out

def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                learning_rate=0.001,
                batch_size=1,
                block_size=20,
                num_epochs=500,
                plot_every=10,
                momentum=0.9,
                dampening=0,
                device='cpu',
                plot=True):
   
 

     
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=dampening)
    
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    for e in range(num_epochs):
        num_iters = 500
        for s in range(num_iters):

            # Get a random slice from our training
            i_start = random.randint(0, len(train_data) - block_size)
            i_end = i_start + block_size + 1

            sequence = train_data[i_start: i_end - 1]
            targets = train_data[i_start+1 : i_end]
            sequence = torch.tensor(sequence)
            targets = torch.tensor(targets)
            # Convert sequences to tensor
            # print("SEQUENCE:", sequence.shape)
            # print("TARGETS:", targets.shape)

            sequence = sequence.to(device)
            targets = targets.to(device)
            # x, t = get_batch(train_data, block_size=block_size, batch_size=batch_size, device=device)

            model.reset_hidden()
            model.zero_grad()

            loss = 0
            # print("T:", targets)
            # We want to minimize the repetitions
            prev_pred = None
            repeats = 0
            for i, char in enumerate(sequence):
                output = model(char)
                # print(output)
                # print(targets[i])
                pred = torch.argmax(output, dim=-1).unsqueeze(0)
                if(pred == prev_pred):
                    repeats += 1
                else:
                    repeats = 0
                prev_pred = pred

                repeat_loss = 2**(repeats)
                # Remove the last repeat term if we are repeating again to only keep this new term
                if repeats > 2:
                    repeat_loss -= 2**(repeats-1)
                
                loss += criterion(output, targets[i].unsqueeze(0)) + repeat_loss



            l1_regularization = 0
            
            # for param in model.parameters():
            #     l1_regularization += torch.norm(param, 1)**2
              
            # loss += l1_regularization
            
            loss.backward()
            optimizer.step()
            
            loss_per_seq = loss.item() / block_size 


            # for i, char in enumerate(sequence):
            #     output = model(char)
            #     # Note: You may need to adjust the shape of the target tensor to match the output
            #     # For example, if targets[i] is a scalar index, you can use torch.tensor([targets[i]])
            #     # targets[i] = targets[i].unsqueeze(0)  # Adjust this line based on the shape of targets
            #     loss += criterion(output, targets[i])

            # # Calculate average loss
            # average_loss = loss / len(sequence)

            # # Backpropagation
            # average_loss.backward()
            # optimizer.step()

        
            # z = model(x)
  
           # pred = torch.argmax(z, dim=1).unsqueeze(1)
    
            # loss = criterion(z, t)
            # l1_regularization = 0
            
           

            iter_count += 1
            if iter_count % plot_every == 0:
                iters.append(iter_count)
                # ta = accuracy(model, train_data, device=device)
                # va = accuracy(model, val_data, device=device)
                train_loss.append(float(loss_per_seq))
                # train_acc.append(ta)
                # val_acc.append(va)
                print("[" + str(iter_count) + "]:","Epoch:",e+1, "Iteration:", s+1, "Loss:", float(loss_per_seq))
    if plot:
        plt.figure()
        plt.plot(iters[:len(train_loss)], train_loss)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        # plt.figure()
        # plt.plot(iters[:len(train_acc)], train_acc)
        # plt.plot(iters[:len(val_acc)], val_acc)
        # plt.title("Accuracy over iterations")
        # plt.xlabel("Iterations")
        # plt.ylabel("Loss")
        # plt.legend(["Train", "Validation"])
        plt.show()

def get_batch(data, block_size, batch_size, device):
    """
    Return a minibatch of data. This function is not deterministic.
    Calling this function multiple times will result in multiple different
    return values.

    Parameters:
        data - a numpy array (e.g., created via a call to np.memmap)
        block_size - the length of each sequence
        batch_size - the number of sequences in the batch
        device - the device to place the returned PyTorch tensor

    Returns: A tuple of PyTorch tensors (x, t), where
        x - represents the input tokens, with shape (batch_size, block_size)
        y - represents the target output tokens, with shape (batch_size, block_size)
    # """

    block_size = min(data.shape[0], block_size)
 
    ix = torch.randint(data.shape[0] - block_size, (batch_size,))
 
    x = torch.stack([(data[i:i+block_size-1]) for i in ix])
 
    t = torch.stack([(data[i+1, i+block_size]) for i in ix])
 
    if 'cuda' in device:
 
        x, t = x.pin_memory().to(device, non_blocking=True), t.pin_memory().to(device, non_blocking=True)
    else:
        x, t = x.to(device), t.to(device)
    assert(x.shape[0] == t.shape[0]), "X is not same shape as T"
    return x, t


def generate_song(model, seed, length=64, tempo=120, starter_size=32, device='cpu'):
    model.reset_hidden()

    i_start = random.randint(0, len(seed) - starter_size)
    i_end = i_start + starter_size + 1

    song = seed[i_start: i_end - 1]
    song = torch.tensor(song)
    song = song.to(device)
    # song = torch.squeeze(song, dim=0)
    # print("Song tensor:", torch.is_tensor(song))
    
    for i in range(song.shape[0]-1):
        _ = model(song[i])

    for j in range(length):
        out = model(song[-1]).data.view(-1)
        
        #dist = out / np.sum(out)
        #print(out.shape)
        dist = F.softmax(out, dim=-1).cpu()
        # We can either use  a probability distribution or simply take the maximum
        pred = torch.argmax(dist, dim=-1).unsqueeze(0).to(device)
        song = torch.cat((song, pred), dim=0)


        # dist = np.array(out)
        # dist = dist / np.sum(dist) # to fix a known np issue
        # pred = np.random.choice(len(dist), p=dist)
        # # Add predicted character to string and use as next input        
        # song = torch.cat((song, torch.tensor(pred).unsqueeze(0).to(device)), dim=0)
    
    
    print(song)

    # print(song.shape)
    # while song.shape[0] < length:

    #     next_note = model(song.unsqueeze(dim=0))

    #     print(next_note.shape)
    #     print(song.shape)
    #     song = torch.cat((song, next_note), dim=0)

    exit()

    midi_file = Midi()
    print('goofy ahh')
    print([song])
    for note in song:
        curr_note = Note(note[0].int().item(), note[1].int().item(), 500, 1200)
        
        midi_file.notes.append(curr_note)
    midi_file.export("./generated_song.mid")    