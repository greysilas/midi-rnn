# Testing
from math import floor
import os
import torch
from midi_parser import Midi
from mido import MidiFile
import numpy as np
import torch.optim as optim 
import matplotlib.pyplot as plt
import model
import model2
import model3
import model4
import pickle
import random


#  GLOBALS
SEQUENCE_LENGTH = 200
BATCH_SIZE = 20
MIDI_PATH = "../data/midis/"

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
    # print("get_batch")
    # print("batch size", batch_size)
    # print("data len", len(data))
    block_size = min(data.shape[0], block_size)
    # print("block size", block_size)
    ix = torch.randint(data.shape[0] - block_size, (batch_size,))
    # print(ix)
    # x = torch.stack([torch.tensor((data[i:i+block_size])).unsqueeze(-1) for i in ix])
    x = torch.stack([(data[i:i+block_size]) for i in ix])
    # t = torch.stack([torch.tensor((data[i+1:i+1+block_size])) for i in ix])
    # OR
    t = torch.stack([(data[i+block_size]) for i in ix])
 
    # WHICH ONE?????????????????????? SECOND ONE
    # THANKS. WELCOME.
    if 'cuda' in device:
        # pin arrays x,t, which allows us to move them to GPU asynchronously
        #  (non_blocking=True)
        x, t = x.pin_memory().to(device, non_blocking=True), t.pin_memory().to(device, non_blocking=True)
    else:
        x, t = x.to(device), t.to(device)
    return x, t


if __name__ == '__main__':
    dir = os.fsencode("../data/midis")
    train_data = []
    val_data = []
    formatted_songs = []
    formatted_notes = []
    total_songs = 0
    train_ratio=0.8
    valid_ratio=0.2

    print("Processing MIDIS")
    files = [f for f in os.listdir(MIDI_PATH) if os.path.isfile(os.path.join(MIDI_PATH, f))]
    random.shuffle(files)

    # Convert all songs into corresponding "Note" object lists format and build dataset
    for filename in files:
        if filename.endswith(".midi") or filename.endswith(".mid"): 
            path = os.path.join(MIDI_PATH, filename)
            curr_midi = Midi(path)
            curr_midi.parse()
            formatted_song = []
            for note in curr_midi.notes:
                formatted_song.append((note.note, note.velocity, note.offset_norm, note.duration_norm))    
            formatted_songs.append(formatted_song)
            formatted_notes.extend(formatted_song)
            total_songs += 1
            if total_songs == 10:
                break
    
        else:
            continue


    # # Serialize the songs
    # with open('../data/serialized/songs2500.pkl', 'wb') as f:  # open a text file
    #     pickle.dump(formatted_songs, f) # serialize the list
    # with open('../data/serialized/notes2500.pkl', 'wb') as f:  # open a text file
    #     pickle.dump(formatted_notes, f) # serialize the list
    

    print("Loading data...")
    # with open('../data/serialized/songs2500.pkl', 'rb') as f:  # open a text file
    #     formatted_songs = pickle.load(f) # serialize the list
    with open('../data/serialized/notes500.pkl', 'rb') as f:  # open a text file
        formatted_notes = pickle.load(f) # serialize the list
    print("Data loaded.")
      

    # Randomly split sequences into training and validation
    random.seed(42)
    random.shuffle(formatted_notes)
    train_data = formatted_notes[:floor(len(formatted_notes)*train_ratio)] 
    val_data = formatted_notes[floor(len(formatted_notes)*train_ratio):]

    device = 'cuda' if torch.cuda.is_available()  else 'cpu'
    device = 'cpu'
    model_type = 4 # Choose model 1, 2, 3
    
    if model_type == 1:
        
        train_data = torch.tensor(train_data)
        val_data = torch.tensor(val_data)
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        
        mod = model.midiRNN1()

        trainModel = False
        model_state_name = './model1weights'
        if trainModel:
            mod.train()
            model.train_model(mod, train_data, num_epochs=1, batch_size=128, block_size=512,
                            plot_every=1, device=device)
            torch.save(mod.state_dict(), model_state_name)
        else:
            mod.load_state_dict(torch.load(model_state_name))
            mod.eval()
            torch.no_grad()
            model.generate_song(mod, val_data, device=device)

    elif model_type == 2:
        dir = os.fsencode("../data/midis")
        train_data = []
        val_data = []
        formatted_songs = []
        formatted_notes = []
        total_songs = 0
        train_ratio=0.8
        valid_ratio=0.2

        print("Processing MIDIS")
        # files = [f for f in os.listdir(MIDI_PATH) if os.path.isfile(os.path.join(MIDI_PATH, f))]
        # random.shuffle(files)

        # for filename in files:
        #     # print(filename)
        #     if filename.endswith(".midi") or filename.endswith(".mid"): 
        #         path = os.path.join(MIDI_PATH, filename) # if below breaks
        #         curr_midi = Midi(path)
        #         curr_midi.parse()
        #         formatted_song = []
        #         for note in curr_midi.notes:
        #             formatted_song.append((note.note, note.velocity, note.offset_norm, note.duration_norm))    
        #         formatted_songs.append(formatted_song)
        #         formatted_notes.extend(formatted_song)
        #         total_songs += 1
        #         if total_songs == 2500:
        #             break
        
        #     else:
        #         continue


        # # Serialize the songs
        # with open('../data/serialized/songs2500.pkl', 'wb') as f:  # open a text file
        #     pickle.dump(formatted_songs, f) # serialize the list
        # with open('../data/serialized/notes2500.pkl', 'wb') as f:  # open a text file
        #     pickle.dump(formatted_notes, f) # serialize the list
        
    
        # exit()
    
        print("Loading data...")
        # with open('../data/serialized/songs2500.pkl', 'rb') as f:  # open a text file
        #     formatted_songs = pickle.load(f) # serialize the list
        with open('../data/serialized/notes500.pkl', 'rb') as f:  # open a text file
            formatted_notes = pickle.load(f) # serialize the list
        print("Data loaded.")
    


        # print("Formatted Notes:", len(formatted_notes))

        # Format data from object to tensor-able list
        # for song in all_songs:
            

        # Randomly split sequences into training and validation
        # random.seed(42)
        # random.shuffle(formatted_songs)

        train_data = formatted_notes[:floor(len(formatted_notes)*train_ratio)] 
        val_data = formatted_notes[floor(len(formatted_notes)*train_ratio):]
        # print("Train data len:" ,len(train_data))
        train_data = [a[0] for a in train_data]
        val_data = [a[0] for a in val_data]
        # train_data = [(float(a[0]),) for a in train_data]
        # val_data = [(float(a[0]),) for a in val_data]
        # print(train_data[-5:])
        #dataset= torch.utils.data.MyIterableDataset(formatted_songs)
        # exit()
        # Concatenate all songs in training set
        # Create model instance

        # Train model on training data

        #print(len(formatted_songs[0]))
        #print(formatted_songs)
        device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        # device = 'cpu'
        

        # print(train_data[:10])
        train_data = torch.tensor(train_data)
        val_data = torch.tensor(val_data)
        
        train_one_hot = torch.nn.functional.one_hot(train_data, num_classes=128).float()
        val_one_hot = torch.nn.functional.one_hot(val_data, num_classes=128).float()
        # # Calculate the mean and standard deviation of each feature in the training set
        # X_mean = train_data.mean(dim=0)
        # X_std = train_data.std(dim=0)

        # # Standardize the training set
        # train_data = (train_data - X_mean) / X_std

        # print(train_data.shape)

        # # Standardize the test set using the mean and standard deviation of the training set
        # val_data = (val_data - X_mean) / X_std
        
        # exit()
    
    
        # train_dataloader = torch.utils.data.DataLoader(formatted_songs, batch_size=10, shuffle=True,
        #                           collate_fn=collate_batch)
        # print(torch.cuda.get_device_name())
        mod = model.midiRNN(128, 256, 128, device=device)
        mod = mod.to(device)
        # train_data = train_data.to(device)
        # val_data = val_data.to(device)
        trainModel = False
        model_state_name = './2500songshidden256epoch5bat128block256'
        if trainModel:
            mod.train()
            model.train_model(mod, train_one_hot, val_one_hot, num_epochs=1, batch_size=128, block_size=512,
                            plot_every=500, device=device)
            torch.save(mod.state_dict(), model_state_name)
        else:
            mod.load_state_dict(torch.load(model_state_name))
            mod.eval()
            torch.no_grad()
            model.generate_song(mod, val_one_hot, device=device)

    
    elif model_type == 3:
        dir = os.fsencode("../data/midis")
        train_data = []
        val_data = []
        formatted_songs = []
        formatted_notes = []
        total_songs = 0
        train_ratio=0.8
        valid_ratio=0.2

        print("Processing MIDIS")
    
    
        print("Loading data...")
        # with open('../data/serialized/songs2500.pkl', 'rb') as f:  # open a text file
        #     formatted_songs = pickle.load(f) # serialize the list
        with open('../data/serialized/notes500.pkl', 'rb') as f:  # open a text file
            formatted_notes = pickle.load(f) # serialize the list
        print("Data loaded.")
    


        # print("Formatted Notes:", len(formatted_notes))

        # Format data from object to tensor-able list
        # for song in all_songs:
            

        # Randomly split sequences into training and validation
        # random.seed(42)
        # random.shuffle(formatted_songs)

        train_data = formatted_notes[:floor(len(formatted_notes)*train_ratio)] 
        val_data = formatted_notes[floor(len(formatted_notes)*train_ratio):]
        # print("Train data len:" ,len(train_data))
        train_data = [a[0] for a in train_data]
        val_data = [a[0] for a in val_data]
        # train_data = [(float(a[0]),) for a in train_data]
        # val_data = [(float(a[0]),) for a in val_data]
        # print(train_data[-5:])
        #dataset= torch.utils.data.MyIterableDataset(formatted_songs)
        # exit()
        # Concatenate all songs in training set
        # Create model instance

        # Train model on training data

        #print(len(formatted_songs[0]))
        #print(formatted_songs)
        device = 'cuda' if torch.cuda.is_available()  else 'cpu'
        # device = 'cpu'
        

        # print(train_data[:10])
        train_data = torch.tensor(train_data)
        val_data = torch.tensor(val_data)
        
        train_one_hot = torch.nn.functional.one_hot(train_data, num_classes=128).float()
        val_one_hot = torch.nn.functional.one_hot(val_data, num_classes=128).float()
        # # Calculate the mean and standard deviation of each feature in the training set
        # X_mean = train_data.mean(dim=0)
        # X_std = train_data.std(dim=0)

        # # Standardize the training set
        # train_data = (train_data - X_mean) / X_std

        # print(train_data.shape)

        # # Standardize the test set using the mean and standard deviation of the training set
        # val_data = (val_data - X_mean) / X_std
        
        # exit()
    
    
        # train_dataloader = torch.utils.data.DataLoader(formatted_songs, batch_size=10, shuffle=True,
        #                           collate_fn=collate_batch)
        # print(torch.cuda.get_device_name())
        mod = model3.midiRNN3() #128, 256, 128, device=device
        mod = mod.to(device)
        # train_data = train_data.to(device)
        # val_data = val_data.to(device)
        trainModel = False
        model_state_name = './model3weights'
        if trainModel:
            mod.train()
            model3.train_model(mod, train_one_hot, num_epochs=1, batch_size=128, block_size=512,
                            plot_every=5, device=device)
            torch.save(mod.state_dict(), model_state_name)
        else:
            mod.load_state_dict(torch.load(model_state_name))
            mod.eval()
            torch.no_grad()
            model3.generate_song(mod, val_one_hot, device=device)
    




    elif model_type == 4:
        dir = os.fsencode("../data/midis")
        train_data = []
        val_data = []
        formatted_songs = []
        formatted_notes = []
        total_songs = 0
        train_ratio=0.8
        valid_ratio=0.2

        print("Processing MIDIS")
    
    
        print("Loading data...")
        # with open('../data/serialized/songs2500.pkl', 'rb') as f:  # open a text file
        #     formatted_songs = pickle.load(f) # serialize the list
        with open('../data/serialized/notes500.pkl', 'rb') as f:  # open a text file
            formatted_notes = pickle.load(f) # serialize the list
        print("Data loaded.")
    


        # print("Formatted Notes:", len(formatted_notes))

        # Format data from object to tensor-able list
        # for song in all_songs:
            

        # Randomly split sequences into training and validation
        # random.seed(42)
        # random.shuffle(formatted_songs)

        train_data = formatted_notes[:floor(len(formatted_notes)*train_ratio)] 
        val_data = formatted_notes[floor(len(formatted_notes)*train_ratio):]
        # print("Train data len:" ,len(train_data))
        train_data = [a[0] for a in train_data]
        val_data = [a[0] for a in val_data]
        # train_data = [(float(a[0]),) for a in train_data]
        # val_data = [(float(a[0]),) for a in val_data]
        # print(train_data[-5:])
        #dataset= torch.utils.data.MyIterableDataset(formatted_songs)
        # exit()
        # Concatenate all songs in training set
        # Create model instance

        # Train model on training data

        #print(len(formatted_songs[0]))
        #print(formatted_songs)
        device = 'cuda' if torch.cuda.is_available()  else 'cpu'
       # device='cpu'
        # device = 'cpu'
        

        # print(train_data[:10])
       # train_data = torch.tensor(train_data)
        #val_data = torch.tensor(val_data)
        
        # train_one_hot = torch.nn.functional.one_hot(train_data, num_classes=128).float()
        # val_one_hot = torch.nn.functional.one_hot(val_data, num_classes=128).float()
        # # # Calculate the mean and standard deviation of each feature in the training set
        # X_mean = train_data.mean(dim=0)
        # X_std = train_data.std(dim=0)

        # # Standardize the training set
        # train_data = (train_data - X_mean) / X_std

        # print(train_data.shape)

        # # Standardize the test set using the mean and standard deviation of the training set
        # val_data = (val_data - X_mean) / X_std
        
        # exit()
    
    
        # train_dataloader = torch.utils.data.DataLoader(formatted_songs, batch_size=10, shuffle=True,
        #                           collate_fn=collate_batch)
        # print(torch.cuda.get_device_name())
        mod = model4.midiRNN4(device=device) #128, 256, 128, device=device
        mod = mod.to(device)
        # train_data = train_data.to(device)
        # val_data = val_data.to(device)
        trainModel = False
        model_state_name = './model4weights'
        if trainModel:
            mod.train()
            model4.train_model(mod, train_data, num_epochs=1, batch_size=25, block_size=25,
                            plot_every=100, device=device, plot=False)
            torch.save(mod.state_dict(), model_state_name)
        else:
            mod.load_state_dict(torch.load(model_state_name))
            mod.eval()
            torch.no_grad()
            model4.generate_song(mod, val_data, device=device)
    