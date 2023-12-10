from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick
import os
from copy import deepcopy

class Note:
    def __init__(self, note, velocity, offset):
        self.note = note
        self.velocity = velocity
        self.offset = offset
        self.duration = None
        # Original Duration = Normalized Duration * (Max Duration − Min Duration) + Min Duration
        self.duration_norm = None   # Normalized duration
        self.offset_norm = None     # Normalized offset

    def __str__(self) -> str:
        return "(Note: " +  str(self.note) + " Velocity: " + str(self.velocity) + " Offset: " + str(self.offset) + " Duration: " + str(self.duration ) + ")"
    
    def __repr__(self) -> str:
        return self.__str__()


class Midi:
    def __init__(self, path=None) -> None:
        self.path = path
        self.notes = []
        if path:
            self.midi = MidiFile(path, clip=True)
        self.min_duration = float('inf')
        self.max_duration = float('-inf')
        self.min_offset = float('inf')
        self.max_offset = float('-inf')
        
    def update_duration_range(self, duration):
        # Set min and max duration for duration normalization
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)

    def update_offset_range(self, offset):
        # Set min and max offset for duration normalization
        self.min_offset = min(self.min_offset, offset)
        self.max_offset = max(self.max_offset, offset)

    def parse(self):
        # Parse tempo
        # metadata_track = self.midi.tracks[0]
        # tempo = 0

        # for i in range(len(metadata_track)):
        #     if metadata_track[i].type == 'set_tempo':
        #         tempo = metadata_track[i].tempo
            
        # quater_note = tempo / 4000 # length of quarter_note
            
        track = self.midi.tracks[1]
        curr_duration = 0
        offset_compensation = 0
        for i in range(len(track)):
            curr_duration = 0

            if track[i].type == 'control_change' or track[i].type == 'note_off' or (track[i].type == 'note_on' and track[i].velocity == 0):
                offset_compensation += track[i].time

            if track[i].type == 'note_on' and track[i].velocity != 0:
                # seq_note = Note(track[i].note, track[i].velocity, 0 if not (track[i].time + offset_compensation) else quater_note)
                seq_note = Note(track[i].note, track[i].velocity, track[i].time + offset_compensation) # If we wanna use this 
                self.notes.append(seq_note)
                self.update_offset_range(track[i].time + offset_compensation)
                offset_compensation = 0

                for j in range(i + 1, len(track)):
                    curr_note = track[j]
                    curr_duration += curr_note.time

                    if (curr_note.type == 'note_off' or (curr_note.type == 'note_on' and curr_note.velocity == 0)) and curr_note.note == seq_note.note:
                        # seq_note.offset += curr_note.time
                        seq_note.duration = curr_duration
                        self.update_duration_range(curr_duration)
                        break

        for note in self.notes:
            # Normalized Offset = (Max Offset − Min Offset) / (Original Offset − Min Offset)
            # Normalized Duration = (Max Duration − Min Duration) / (Original Duration − Min Duration) 
            # note.offset_norm = (note.offset - self.min_offset) / (self.max_offset - self.min_offset)  # adding epsilon
            note.offset_norm = (note.offset - self.min_offset) / max(self.max_offset - self.min_offset, 1e-8)
            # note.duration_norm = (self.max_duration - self.min_duration) / (note.duration - self.min_duration + 1e-8)
            note.duration_norm = (note.duration - self.min_duration) / max(self.max_duration - self.min_duration, 1e-8)
            
        return self.notes
    
    def export(self, path):
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        notes = deepcopy(self.notes)
        start = 0
        for i  in range(len(notes)): #len(notes)
            # Look at current note
            curr_note = notes[i]
            played_note_compensation = 0
            # Notes which finish before current note
            finished_notes = []
            # Looking at previous notes
            for j in range(start, i):
                # This note is already played
                if all(n is None for n in notes[start:j]):
                    start = j
                if notes[j] == None:
                    True
                # This note should end before current note
                elif notes[j].duration - curr_note.offset <= 0:
                    finished_notes.append(j)
                # Note won't end yet, adjust duration accordingly
                else:
                    notes[j].duration -= curr_note.offset
            # Sort finished notes by remaining duration
            finished_notes.sort(key = lambda x: notes[x].duration)
            for n in finished_notes:
                time = notes[n].duration - played_note_compensation
                track.append(Message('note_on', note=notes[n].note, velocity=0, time=time))
                # Compensate current note's offset for new note_ends inserted
                played_note_compensation += time
                # Set notes as played
                notes[n] = None
            # Add current note
            track.append(Message('note_on', note=curr_note.note, velocity=curr_note.velocity, time=curr_note.offset-played_note_compensation))

        track.append(MetaMessage('end_of_track', time=1))
        midi.save(path)


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
# mid = Midi('../data/midis/Satie, Erik, 3 Gymnopédies, _fuIMye31Gw.mid')
mid = Midi('../data/midis/Beethoven, Ludwig van, Für Elise, WoO 59, noAU3qDS1dA.mid')
mid.parse()
# # for note in mid.notes:
# #     print(note)
mid.export("export_test.mid")
