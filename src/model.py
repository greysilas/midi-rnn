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
        # self.fc0 = nn.Linear(input_size, hidden_size)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        # self.fc2 = nn.Linear(hidden_size,output_size ) 
      
    def forward(self, X):
        # print("X shape before fc0:", X.shape)
        # out = self.fc0(X.float())
        # print("X shape", X.shape)
        out, h = self.rnn(X.float(),) # (block_size, hidden_size)

        # print("OUT shape:", out.shape)
        # features = torch.cat([torch.amax(h, dim=1),
        #                       torch.mean(h, dim=1)], axis=-1)
        out = self.fc1(out[:, -1, :]) # Gets the last timestep
       # out = F.softmax(out)

        # print("Out prob dist: ", out)
        # print("Highest prob: ", torch.max(out))

        # print("FORWARD Z shape:",z.shape)
        # y = self.activation(z) #(hidden_size, output_size)
        # print("Y shape:",y.shape)
        return out

def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.0003,
                batch_size=1,
                block_size=20,
                num_epochs=500,
                plot_every=10,
                device='cpu',
                plot=True):
   

    # Convert input to one hots
    # train_one_hots = []
    # for note in train_data:
    #     x_one_hot = np.zeros(128)
    #     x_one_hot[note ] = 1

   

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    for e in range(num_epochs):
        num_batches = 2500
        for s in range(num_batches):
            x, t = get_batch(train_data, block_size=block_size, batch_size=batch_size, device=device)
            
            # print(t)
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
            pred = torch.argmax(z, dim=1).unsqueeze(1)
            # print("Z", z)
            # print("pred", pred)
            # print("T", t)
            # print("Pred shape", pred.shape)
            # print("T Shape", t.shape)
            # print("Pred shape", pred.shape)
            # print("z shape", z.shape)
            # print(z)
            # print(t.squeeze().long())
            # one_hot_t = torch.nn.functional.one_hot(t.squeeze().long(), num_classes=128).float()

            # print("one hot t shape", one_hot_t.shape)
            # print(one_hot_t)
            loss = criterion(z, t)
            l1_regularization = 0
            
            for param in model.parameters():
                l1_regularization += torch.norm(param, 1)**2
              
            loss += l1_regularization

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iter_count += 1
            if iter_count % plot_every == 0:
                iters.append(iter_count)
                # ta = accuracy(model, train_data, device=device)
                # va = accuracy(model, val_data, device=device)
                train_loss.append(float(loss))
                # train_acc.append(ta)
                # val_acc.append(va)
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

def generate_song(model, seed, length=64, tempo=120, starter_size=32, device='cpu'):
    # print("Generate song")
    song, t = get_batch(seed, block_size=starter_size, batch_size=1, device=device)
    song = song.squeeze(dim=0)
    # song = torch.squeeze(song, dim=0)
    # print("Song tensor:", torch.is_tensor(song))

    # print(song.shape)
    while song.shape[0] < length:
        # print("Current sequence", torch.unsqueeze(song, dim=0)[-20:])
        next_note = model(song[-20:].unsqueeze(dim=0))
        # torch.sum(probs,dim=1)
        # print(song.shape)
        # print(probs)
        # print("Note tensor:", torch.is_tensor(next_note))
        # print(next_note.shape)
        print(next_note.shape)
        print(song.shape)
        song = torch.cat((song, next_note), dim=0)
        # print("New song shape", song.shape)
        # Sample the next note (you might need to adjust this depending on your output space)
        # next_note = torch.multinomial(F.softmax(next_note, dim=-1).view(-1), 1).view(1, -1)
        # print(song)
    # Convert to note
    midi_file = Midi()
    print('goofy ahh')
    print([song])
    for note in song:
        # Create note object
        
        # curr_note = Note()
        # Denormalize
        # Add it to song

        True
    # midi_file.export("./generated_song.mid")
    # print(song)
