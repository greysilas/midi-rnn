from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
import os
from copy import deepcopy

class Note:
    def __init__(self, note, velocity, offset):
        self.note = note
        self.velocity = velocity
        self.offset = offset
        self.duration = None

    def set_duration(self, duration):
        self.duration = duration
    
    def __str__(self) -> str:
        return "(Note: " +  str(self.note) + " Velocity: " + str(self.velocity) + " Offset: " + str(self.offset) + " Duration: " + str(self.duration ) + ")"
    
    def __repr__(self) -> str:
        return self.__str__()


class Midi:
    def __init__(self, path) -> None:
        self.path = path
        self.notes = []
        self.midi = MidiFile(path, clip=True)
        
    def parse(self):
        track = self.midi.tracks[1]
        curr_duration = 0
        sequence = []
        offset_compensation = 0
        for i in range(len(track)):
            curr_duration = 0
            
            if track[i].type == 'control_change' or track[i].type == 'note_off' or (track[i].type == 'note_on' and track[i].velocity == 0):
                offset_compensation += track[i].time

            if track[i].type == 'note_on' and track[i].velocity != 0:
                seq_note = Note(track[i].note, track[i].velocity, track[i].time + offset_compensation)
                offset_compensation = 0
                sequence.append(seq_note)

                for j in range(i + 1, len(track)):
                    curr_note = track[j]
                    curr_duration += curr_note.time

                    if (curr_note.type == 'note_off' or (curr_note.type == 'note_on' and curr_note.velocity == 0)) and curr_note.note == seq_note.note:
                        # seq_note.offset += curr_note.time
                        seq_note.set_duration(curr_duration)
                        break
        self.notes = sequence.copy()
    
    def export(self, path):
        # TODO: Figure out time
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        played_note_compensation = 0
        notes = deepcopy(self.notes)
        start = 0
        for i  in range(15): #len(notes)
            # Look at current note
            curr_note = notes[i]
            played_note_compensation = 0
            if curr_note.note == 64 and curr_note.velocity==64:
                True
            # print("Current Note:", notes[i])
            # End any notes before current note
            for j in range(start, i):
                # Look at previous notes
                # If note at index == None, already done playing the note

                finished_notes = []
                if notes[j] == None:
                    True
                # Subtract current offset from their duration
                elif notes[j].duration - curr_note.offset <= 0:
                    # End the note
                    time = notes[j].duration - played_note_compensation
                    # TODO:
                    # Problem: Two or more notes could end after the same note,
                    #       but they would have different notations. If shorter
                    #       note is added later, it's time would be negative.
                    # Fix: Store finished notes somewhere and insert by duration left
                    # finished_notes.append(notes[j])
                    track.append(Message('note_on', note=notes[j].note, velocity=0, time=time))
                    # Store this value to compensate in the next "played note"
                    played_note_compensation += time
                    # Set the note as played
                    notes[j] = None
                    if j == start:
                        start += 1
                else:
                    notes[j].duration -= curr_note.offset
            # Add current note
            track.append(Message('note_on', note=curr_note.note, velocity=curr_note.velocity, time=curr_note.offset-played_note_compensation))
            
        track.append(MetaMessage('end_of_track', time=1))
        # print(track)
        # midi.save(path)



    def add_note(self, note: Note):
        pass



# mid = MidiFile('../data/midis/Beethoven, Ludwig van, Für Elise, WoO 59, noAU3qDS1dA.mid', clip=True)
# print(mid.tracks[1])
# track = mid.tracks[1]
# track.pop()
# track.append(Message('note_on', channel=0, note=60, velocity=64, time=0))
# track.append(MetaMessage('end_of_track'))
# print(mid.tracks[1])
# mid.save('new_song.mid')

mid = Midi('../data/midis/Beethoven, Ludwig van, Für Elise, WoO 59, noAU3qDS1dA.mid')
mid.parse()
# for note in mid.notes:
#     print(note)
mid.export("export_test.mid")