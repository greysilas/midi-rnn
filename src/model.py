import numpy
import pandas
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F
from midi_parser import Midi, Note

# HYPERPARAMETERS
INPUT_SIZE = 4
HIDDEN_SIZE = 128
OUTPUT_SIZE = 4

class midiRNN1(nn.Module):
    def __init__(self):
        super(midiRNN1, self).__init__()
        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.output_size = OUTPUT_SIZE
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
         
      
    def forward(self, X):
        out, h = self.rnn(X.float(),) # (block_size, hidden_size)
        out = self.fc1(out[:, -1, :]) # Gets the last timestep
        return out
    
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

    x = torch.stack([(data[i:i+block_size]) for i in ix])

    t = torch.stack([(data[i+block_size]) for i in ix])

    if 'cuda' in device:

        x, t = x.pin_memory().to(device, non_blocking=True), t.pin_memory().to(device, non_blocking=True)
    else:
        x, t = x.to(device), t.to(device)
    return x, t


def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                learning_rate=0.0003,
                batch_size=1,
                block_size=20,
                num_epochs=500,
                plot_every=10,
                device='cpu',
                plot=True):
   


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    for e in range(num_epochs):
        num_batches = 2500
        for s in range(num_batches):
            x, t = get_batch(train_data, block_size=block_size, batch_size=batch_size, device=device)

            z = model(x)

            pred = torch.argmax(z, dim=1).unsqueeze(1)

            loss = criterion(z, t)
            l1_regularization = 0
            
            for param in model.parameters():
                l1_regularization += torch.norm(param, 1)**2
              
            loss += l1_regularization

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iter_count += 1
            if iter_count % plot_every == 0:
                iters.append(iter_count)

                train_loss.append(float(loss))

                print("[" + str(iter_count) + "]:","Epoch:",e+1, "Iteration:", s+1, "Loss:", float(loss))
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
        # plt.figure()
        # plt.plot(iters[:len(train_acc)], train_acc)
        # plt.plot(iters[:len(val_acc)], val_acc)
        # plt.title("Accuracy over iterations")
        # plt.xlabel("Iterations")
        # plt.ylabel("Loss")
        # plt.legend(["Train", "Validation"])
        plt.show()

def generate_song(model, seed, length=64, tempo=120, starter_size=32, device='cpu'):
    # print("Generate song")
    song, t = get_batch(seed, block_size=starter_size, batch_size=1, device=device)
    song = song.squeeze(dim=0)
    # song = torch.squeeze(song, dim=0)
    song = song.squeeze(dim=0)
    # song = torch.squeeze(song, dim=0)
    # print("Song tensor:", torch.is_tensor(song))

    # print(song.shape)
    while song.shape[0] < length:

        next_note = model(song[-20:].unsqueeze(dim=0))

        print(next_note.shape)
        print(song.shape)
        song = torch.cat((song, next_note), dim=0)

    midi_file = Midi()
    print('goofy ahh')
    print([song])
    print('goofy ahh')
    print([song])
    for note in song:
        # Original Duration = Normalized Duration * (Max Duration − Min Duration) + Min Duration
        original_duration = note[2] * 120
        original_offset = note[3] * 120
        curr_note = Note(note[0].int().item(), note[1].int().item(), original_offset.float().item(), original_duration.float().item())
        
        midi_file.notes.append(curr_note)
    midi_file.export("./generated_song.mid")    