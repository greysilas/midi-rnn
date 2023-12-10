import numpy
import pandas
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F
from main import get_batch, BATCH_SIZE
from midi_parser import Midi

# MODEL PARAMETERS
INPUT_SIZE = None
HIDDEN_SIZE = 128
OUTPUT_SIZE = None

class midiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(midiRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, 2*hidden_size, batch_first=True)
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 4) 
        # self.activation = nn.Tanh()

    def forward(self, X):
        # print(X.shape)
        out, h = self.rnn(X) # (block_size, hidden_size)
      
        # print("OUT shape:", out.shape)
        # features = torch.cat([torch.amax(h, dim=1),
        #                       torch.mean(h, dim=1)], axis=-1)
        out = F.relu(self.fc1(out[:, -1, :])) # Gets the last timestep
        out = F.relu(self.fc2(out))

        # print("FORWARD Z shape:",z.shape)
        # y = self.activation(z) #(hidden_size, output_size)
        # print("Y shape:",y.shape)
        return out

def accuracy(model, dataset, max=1000, device='cpu'):
    correct, total = 0, 0
    # dataloader = DataLoader(dataset,
    #                         batch_size=1,  # use batch size 1 to prevent padding
    #                         collate_fn=collate_batch)
    #for i, (x, t) in enumerate(dataloader):
    #     z = model(x)
    #     y = torch.argmax(z, axis=)
    # for each song in dataset:
    #   for i in range(num_batches): same number as hyperparameter set in main.py
    #       get random batch (get_batch), 
    #           Get prediction, +1 if correct
    #
    # print("IN ACCURACY")
    # print("Validation len", len(dataset))
    for i in range(BATCH_SIZE):
        xs, ts = get_batch(dataset, batch_size=100, block_size=256,device=device)
        y = model(xs)
        o = torch.eq(y, ts)
        # print("Y shape", y.shape)
        # print("Ts shape", ts.shape)
        # print(y[:2])
        # print(ts[:2])
        if False not in o:
            correct += 1
        total +=1
    return correct / total

def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.001,
                batch_size=1,
                block_size=256,
                num_epochs=500,
                num_iters_per_score = 5,
                plot_every=10,
                device='cpu',
                plot=True):
   
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    for e in range(num_epochs):
        x, t = get_batch(train_data, block_size=block_size, batch_size=batch_size, device=device)
    # for i in range(xs.shape[0]):
        # print(len(get_batch(train_data, block_size=block_size, batch_size=batch_size,device=device)[0]))
    #     x = xs[i]
    #     t = ts[i]  
        z = model(x)
        # print("Z")
        # print(z)
        # print("T")
        # print(t)
        
        # print(z)
        # print(ts)
        # print("Z:", z.shape)
        # print("Ts", t.shape)
        loss = criterion(z, t)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        iter_count += 1
        if iter_count % plot_every == 0:
            iters.append(iter_count)
            ta = accuracy(model, train_data, device=device)
            va = accuracy(model, val_data, device=device)
            train_loss.append(float(loss))
            train_acc.append(ta)
            val_acc.append(va)
            print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
    if plot:
        plt.figure()
        plt.plot(iters[:len(train_loss)], train_loss)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        plt.figure()
        plt.plot(iters[:len(train_acc)], train_acc)
        plt.plot(iters[:len(val_acc)], val_acc)
        plt.title("Accuracy over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"])

def generate_song(model, seed, length=64, tempo=120, starter_size=32, device='cpu'):
    # print("Generate song")
    song, t = get_batch(seed, block_size=starter_size, batch_size=1, device=device)
    song = torch.squeeze(song, dim=0)
    # print("Song tensor:", torch.is_tensor(song))

    # print(song.shape)
    while len(song) < length:
        next_note = model(torch.unsqueeze(song, dim=0))
        # print("Note tensor:", torch.is_tensor(next_note))
        # print(next_note.shape)
        song = torch.cat((song, next_note), dim=0)
        # print("New song shape", song.shape)
        # Sample the next note (you might need to adjust this depending on your output space)
        # next_note = torch.multinomial(F.softmax(next_note, dim=-1).view(-1), 1).view(1, -1)
        # print(song)
    # Convert to note
    midi_file = Midi()
    print(song)
    print(song.shape)
    for note in song:
        # Create note object
        
        # curr_note = Note()
        # Denormalize
        # Add it to song

        True
    # midi_file.export("./generated_song.mid")
    # print(song)
