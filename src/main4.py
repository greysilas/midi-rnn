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
import math


#  GLOBALS
SEQUENCE_LENGTH = 200
BATCH_SIZE = 20
MIDI_PATH = "../data/midis/"
def round_to_nearest(x, base=5):

   return base * math.ceil(x / base)

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
    #         if total_songs % 25 == 0:
    #             print("Processed:",total_songs)
    #         if total_songs == 250:
    #             break
    
    #     else:
    #         continue


    # # Serialize the songs
    # with open('../data/serialized/songs250.pkl', 'wb') as f:  # open a text file
    #     pickle.dump(formatted_songs, f) # serialize the list
    # with open('../data/serialized/notes250.pkl', 'wb') as f:  # open a text file
    #     pickle.dump(formatted_notes, f) # serialize the list
    

    #     exit()
    







    print("Loading data...")
    # with open('../data/serialized/songs2500.pkl', 'rb') as f:  # open a text file
    #     formatted_songs = pickle.load(f) # serialize the list
    with open('../data/serialized/notes2500.pkl', 'rb') as f:  # open a text file
        formatted_notes = pickle.load(f) # serialize the list
    print("Data loaded.")


    
    # Divide the offsets and durations into discrete classes
    # Each class will represent a jump of 100 units 
    # The maximum value for both is clamped to 3000
    # Both are multiplied by 10000 to get easier numbers to work with 
    OFFSET_MAX = 3000
    OFFSET_STEP = 300
    OFFSET_CLASSES = (OFFSET_MAX // OFFSET_STEP) + 1 # We add one since this is 0-based

    DURATION_MAX = 3000
    DURATION_STEP = 300
    DURATION_CLASSES = (DURATION_MAX // DURATION_STEP) + 1

    set_offsets = set()
    set_durations = set()
    for i in range(len(formatted_notes)):
        # Change offset and duration
        offset = formatted_notes[i][2]
        offset = min(3000, round_to_nearest(offset * 10000,300) ) // OFFSET_STEP
        set_offsets.add(offset)
        duration = formatted_notes[i][3]
        duration = min(3000, round_to_nearest(duration * 10000,300) ) // OFFSET_STEP
        set_durations.add(duration)

        formatted_notes[i] = (formatted_notes[i][0], formatted_notes[i][1], offset, duration) 
    
  
 
    train_data = formatted_notes[:floor(len(formatted_notes)*train_ratio)] 
    val_data = formatted_notes[floor(len(formatted_notes)*train_ratio):]
    
    # train_data = [a[0] for a in train_data]
    # val_data = [a[0] for a in val_data]

    device = 'cuda' if torch.cuda.is_available()  else 'cpu'
    # device = 'cpu'

    print("Device:", device)

    mod = model4.midiRNN4(note_classes=128, velocity_classes=128,
                          offset_classes=OFFSET_CLASSES, 
                          duration_classes=DURATION_CLASSES,device=device) #128, 256, 128, device=device
    mod = mod.to(device)
    # train_data = train_data.to(device)
    # val_data = val_data.to(device)
    trainModel = True
    model_state_name = './model4weights'
    if trainModel:
        mod.train()
        model4.train_model(mod, train_data, num_epochs=50, batch_size=25, block_size=25,
                        plot_every=500, device=device, plot=True)
        torch.save(mod.state_dict(), model_state_name)
    else:
     
        mod.load_state_dict(torch.load(model_state_name))
        mod.eval()
        torch.no_grad()
        model4.generate_song(mod, val_data, starter_size=2, 
                            length=128,offset_step=OFFSET_STEP, 
                            duration_step=DURATION_STEP, use_dist_sample=True,
                            device=device)
