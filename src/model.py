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
        print(X.shape)
        out, h = self.rnn(X) # (block_size, hidden_size)
      
        print("OUT shape:", out.shape)
        # features = torch.cat([torch.amax(h, dim=1),
        #                       torch.mean(h, dim=1)], axis=-1)
        z = self.fc(out[:, -1, :]) # Gets the last timestep
        print("FORWARD Z shape:",z.shape)
        # y = self.activation(z) #(hidden_size, output_size)
        # print("Y shape:",y.shape)
        return z

def accuracy(model, dataset, max=1000):
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
    for score in dataset:
        for i in range(BATCH_SIZE):
            xy, ty = get_batch(score, block_size=20, batch_size=100,device='cpu')
            for i in range(xy.shape[0]):
                x = xy[i]
                t = ty[i]
                y = model(x)
                o = torch.eq(y[-1], t)
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
                batch_size=100,
                block_size=20,
                num_epochs=10,
                num_iters_per_score = 5,
                plot_every=10,
                device='cpu',
                plot=True):
   
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    for e in range(num_epochs):
        for score in train_data:
            
            for i in range(num_iters_per_score):

                xs, ts = get_batch(score, block_size=block_size, batch_size=batch_size,device=device)
            # for i in range(xs.shape[0]):
                print(len(get_batch(score, block_size=block_size, batch_size=batch_size,device=device)[0]))
            #     x = xs[i]
            #     t = ts[i]
            
                
                z = model(xs)
               # print(len(z))
                #print(t)]
                
                # print(z)
                # print(ts)
                print("Z:", z.shape)
                print("Ts", ts.shape)
                loss = criterion(z, ts)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data)
                    va = accuracy(model, val_data)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)

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
