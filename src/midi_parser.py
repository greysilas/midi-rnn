from music21 import converter, instrument, note, chord

notes = []
count = 0
count1 = 0
midi = converter.parse("gary_jules-mad_world.mid")
notes_to_parse = None
parts = instrument.partitionByInstrument(midi)
if parts: # file has instrument parts
    notes_to_parse = parts.parts[0].recurse()
else: # file has notes in a flat structure
    notes_to_parse = midi.flat.notes
for element in notes_to_parse:
    count1 += 1
    try:
        print(element.pitches, element.duration)
        count += 1
    except:
        pass
    if isinstance(element, note.Note):
        notes.append(str(element.pitch))
    elif isinstance(element, chord.Chord):
        notes.append('.'.join(str(n) for n in element.normalOrder))

print(count)
print(count1)