from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick


class Note:
    def __init__(self, note, velocity, offset):
        self.note = note
        self.velocity = velocity
        self.offset = offset

    def set_duration(self, duration):
        self.duration = duration
    
    def __str__(self) -> str:
        return "Note: " +  str(self.note) + " Velocity: " + str(self.velocity) + " Offset: " + str(self.offset) + " Duration: " + str(self.duration )


class Midi:
    def __init__(self, path) -> None:
        self.path = path
        self.notes = []
        self.midi = MidiFile(path, clip=True)
        
    def parse(self):
        track = self.midi.tracks[1]
        curr_duration = 0
        sequence = []

        for i in range(len(track)):
            curr_duration = 0
            if track[i].type == 'note_on' and track[i].velocity != 0:
                seq_note = Note(track[i].note, track[i].velocity, track[i].time)
                sequence.append(seq_note)

                for j in range(i + 1, len(track)):
                    curr_note = track[j]
                    curr_duration += curr_note.time

                    if (curr_note.type == 'note_off' or (curr_note.type == 'note_on' and curr_note.velocity == 0)) and curr_note.note == seq_note.note:
                        seq_note.set_duration(curr_duration)
                        break
    
    def export(self, path):
        pass

    def add_note(self, note: Note):
        pass



# mid = MidiFile('../data/midis/Turchetto, Andrea, Variations on a Theme by Mozart, QihjMKKNdo0.mid', clip=True)
# print(mid.tracks[1])
# track = mid.tracks[1]
# track.pop()
# track.append(Message('note_on', channel=0, note=60, velocity=64, time=0))
# track.append(MetaMessage('end_of_track'))
# print(mid.tracks[1])
# mid.save('new_song.mid')