# Testing
from math import floor
import os
import random
import torch
from midi_parser import Midi
from mido import MidiFile
import numpy as np
import torch.optim as optim 
import matplotlib.pyplot as plt
import model
import pickle


#  GLOBALS
SEQUENCE_LENGTH = 200
BATCH_SIZE = 20
MIDI_PATH = "../data/midis/"

def pad_sequence(text_list, padding_value):
    pass

def collate_batch(batch):
    """
    Returns the input and target tensors for a batch of data

    Parameters:
        `batch` - An iterable data structure of tuples (indices, label),
                  where `indices` is a sequence of word indices, and
                  `label` is either 1 or 0.

    Returns: a tuple `(X, t)`, where
        - `X` is a PyTorch tensor of shape (batch_size, sequence_length)
        - `t` is a PyTorch tensor of shape (batch_size)
    where `sequence_length` is the length of the longest sequence in the batch
    """
    text_list = []  # collect each sample's sequence of word indices
    label_list = [] # collect each sample's target labels

    for (note_list, label) in batch:
        text_list.append(torch.tensor(text_indices))
        # TODO: what do we need to do with `label`?
        label_list.append(label)
        
    # song
    # seq_len = 20 <-  
    # for i in range(len(song)-seq_len + 1):
    #   X = song[i:seq_len]
    #   t = song[seq_len]
    #   i++
    #
    
    X = pad_sequence(text_list, padding_value=3).transpose(0, 1)
    t = torch.tensor(label_list) # TODO
    return X, t



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
    block_size = min(len(data), block_size)
    # print("block size", block_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([torch.tensor((data[i:i+block_size])) for i in ix])
    # t = torch.stack([torch.tensor((data[i+1:i+1+block_size])) for i in ix])
    # OR
    t = torch.stack([torch.tensor((data[i+block_size])) for i in ix])
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
    # Convert all songs into corresponding "Note" object lists format and build dataset
    
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
    
    for filename in files:
        # print(filename)
        if filename.endswith(".midi") or filename.endswith(".mid"): 
            path = os.path.join(MIDI_PATH, filename) # if below breaks
            curr_midi = Midi(path)
            curr_midi.parse()
            formatted_song = []
            for note in curr_midi.notes:
                formatted_song.append((note.note, note.velocity, note.offset_norm, note.duration_norm))    
            formatted_songs.append(formatted_song)
            formatted_notes.extend(formatted_song)
            total_songs += 1
       
        else:
            continue


    # Serialize the songs
    with open('../data/serialized/songs10854.pkl', 'wb') as f:  # open a text file
        pickle.dump(formatted_songs, f) # serialize the list
    with open('../data/serialized/notes10854.pkl', 'wb') as f:  # open a text file
        pickle.dump(formatted_notes, f) # serialize the list
    
 
    exit()
   
    with open('../data/serialized/songs500.pkl', 'rb') as f:  # open a text file
        formatted_songs = pickle.load(f) # serialize the list
    with open('../data/serialized/notes500.pkl', 'rb') as f:  # open a text file
        formatted_notes = pickle.load(f) # serialize the list
    
    
    print("Songs:", formatted_songs[0])
    print("Notes:", formatted_notes[0])


    # print("Formatted Notes:", len(formatted_notes))

    # Format data from object to tensor-able list
    # for song in all_songs:
        

    # Randomly split sequences into training and validation
    # random.seed(42)
    # random.shuffle(formatted_songs)

    train_data = formatted_notes[:floor(len(formatted_notes)*train_ratio)] 
    val_data = formatted_notes[floor(len(formatted_notes)*train_ratio):]
 
    #dataset= torch.utils.data.MyIterableDataset(formatted_songs)
    
    # Concatenate all songs in training set
    # Create model instance

    # Train model on training data

    #print(len(formatted_songs[0]))
    #print(formatted_songs)
    device = 'cuda' if torch.cuda.is_available()  else 'cpu'
  
  
    # train_dataloader = torch.utils.data.DataLoader(formatted_songs, batch_size=10, shuffle=True,
    #                           collate_fn=collate_batch)
    print(torch.cuda.get_device_name())
    mod = model.midiRNN(4, 256, 4)
    mod = mod.to(device)
    # mod.train()
    # model.train_model(mod, train_data, val_data, num_epochs=3000, batch_size=128, block_size=256,
    #                    plot_every=250, device=device)
    # torch.save(mod.state_dict(), './500songshidden256epoch3000bat128block256')
    
    
    # mod.load_state_dict(torch.load('./500songshidden256epoch3000bat128block256'))
    # mod.eval()
    # torch.no_grad()
    # model.generate_song(mod, val_data, device=device)
    

    # data = sequence of all notes
    #   - training
    #   - seeds
    # 
    # generation
    #   - get_batch(seed) in seed, extract a random batch
    #   - x number of notes to start, then generate after
