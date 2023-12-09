import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size)

    def forward(self, X):
        out, h = self.rnn(X)
        z = self.fc(out)
        return z

def accuracy(model, dataset, max=1000):
    correct, total = 0, 0
    dataloader = DataLoader(dataset,
                            batch_size=1,  # use batch size 1 to prevent padding
                            collate_fn=collate_batch)
    for i, (x, t) in enumerate(dataloader):
        z = model(x)
        y = torch.argmax(z, axis=1)
        correct += int(torch.sum(t == y))
        total   += 1
        if i >= max:
            break
    return correct / total

def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.001,
                batch_size=100,
                num_epochs=10,
                plot=True):
   
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    for e in range(num_epochs):
        for i, (notes, duration, sequence) in enumerate(train_loader):
             z = model(notes, duration)
             loss = criterion(z, sequence)

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
