from math import floor
import os
import random
import torch
from midi_parser import Midi
# NOTE: Not used
class MidiBatcher:
    def __init__(self, midis, batch_size=32, drop_last=False):
        self.midis_by_length = {}
        for midi in midis:
            midi_len = len(midi)
            if midi_len not in self.midis_by_length:
                self.midis_by_length[midi_len] = []
            self.midis_by_length[midi_len].append(midi)
        self.loaders = {midi_len : torch.utils.data.DataLoader(
                                        midis,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        drop_last = drop_last) 
                        for midi_len, midis in self.midis_by_length.items()}

    def __iter__(self):
        iters = [iter(loader) for loader in self.loaders.values()]
        while iters:
            # pick an iterator (a length)
            im = random.choice(iters)
            try:
                yield next(im)
            except StopIteration:
                # no more elements in the iterator, remove it
                iters.remove(im)


print("running")
print(torch.cuda.is_available())
dir = os.fsencode("../data/midis")
train_data = []
val_data = []
formatted_songs = []
formatted_notes = []
total_songs = 0
train_ratio=0.8
valid_ratio=0.2

MIDI_PATH = '../data/midis/'

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
        print(len(formatted_song))
        formatted_notes.extend(formatted_song)
        total_songs += 1
        if total_songs == 10:
            break
    else:
        continue
print(len(formatted_notes))

    

# Format data from object to tensor-able list
# for song in all_songs:
    

# Randomly split sequences into training and validation

train_data = formatted_notes[:int(total_songs*train_ratio)] 
val_data = formatted_notes[int(total_songs*train_ratio):]
batcher = MidiBatcher(train_data, drop_last=True)
print(len(train_data))
print(len(batcher.loaders))
for i, (song) in enumerate():
    if i > 5: break
    print(song.shape)