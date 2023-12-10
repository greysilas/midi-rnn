import numpy
import pandas
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn
from main import get_batch, BATCH_SIZE

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
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.activation = nn.Tanh()

    def forward(self, X):
        # print(X.shape)
        out, h = self.rnn(X) # (block_size, hidden_size)
      
        # print("OUT shape:", out.shape)
        # features = torch.cat([torch.amax(h, dim=1),
        #                       torch.mean(h, dim=1)], axis=-1)
        z = self.fc(out[:, -1, :]) # Gets the last timestep
        # print("FORWARD Z shape:",z.shape)
        # y = self.activation(z) #(hidden_size, output_size)
        # print("Y shape:",y.shape)
        return z

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
        # print("Y")
        # print(y)
        # print("TS")
        # print(ts)
        # print("X shape", xs.shape)
        # print("Y shape", y.shape)
        # print("T shape", ts.shape)
        #print("y:", y[-1])
        #print("t:", t)
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
        # print(len(z))
        #print(t)]
        
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

def generate_song(model):
    pass
