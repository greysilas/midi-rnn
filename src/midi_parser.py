from music21 import converter, instrument, note, chord, midi



class Midi:

    def __init__(self, path, quantize=True):
        '''
        Initializes the MIDI by opening and storing the file associated with the 
        file path
        '''
        self.path = path

        # Store the score object
        self.score = converter.parse(self.path, quantizePost=quantize)
    
     
    def get_notes(self):
        '''
        Get a list of Note objects
        '''
        pass

    def set_instrument(self,instrument, part=0):
        self.score.parts[part].insert(0, instrument)

    def export(self, output_path):
       self.score.write('midi', output_path)




file_path = '../data/midis/Turchetto, Andrea, Variations on a Theme by Mozart, QihjMKKNdo0.mid'
m = Midi(file_path)
m.set_instrument(instrument.Violin())
m.export('./out_violin.midi')

