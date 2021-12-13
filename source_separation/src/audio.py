from camns_lp import *
import pathlib
import soundfile as sf

class camns_audio(camns_object):
    SAVED = 0
    MAX_SIZE = 300000
    def __init__(self, file: str=None, folder=None, sr=None):
        super().__init__()

        if file is not None:

            if folder is None:
                file = pathlib.Path(__file__).parent.parent.resolve() / 'audio' / file
            else:
                file = pathlib.PurePath(folder, file)

            self.vector__, self.sr = sf.read(file)
            self.vector__ = self.vector__[:camns_audio.MAX_SIZE, :1].reshape(-1,)
            self.size__ = self.vector__[0]
            self.has_audio__ = True
        else:
            self.has_audio__ = False
            self.sr = sr

    def write(self, name=None):
        if name is None:
            name = 'audio{}.wav'.format(camns_audio.SAVED)
            camns_audio.SAVED += 1
        file = pathlib.Path(__file__).parent.parent.resolve() / 'res' / name
        sf.write(str(file), self.vector__, self.sr)

    def set_sounds(self, sounds):

        self.vector__ = np.array(list(sounds))
        self.has_audio__ = True

        return self

    @staticmethod
    def mix(audio, observ_num=None):
        sounds = camns_audio.to_audio_matrix(audio)
        res = get_random_observations(sounds, observ_num)
        mixed = camns_audio.from_audio_matrix(res, audio[0].sr)
        return mixed

    @staticmethod
    def camns_lp(audio, observ_num=None):
        sounds = camns_audio.to_audio_matrix(audio)
        res = camns_lp(sounds, observ_num)
        unmixed = camns_audio.from_audio_matrix(res, audio[0].sr)
        return unmixed

    @staticmethod
    def to_audio_matrix(audio):
        """
        audio = [sound1, sound2, ...]
        output: (L, M) - L - sound size, M - number of sounds
        """
        return np.array(list(map(lambda x: x.vector__, audio))).T

    @staticmethod
    def from_audio_matrix(audio, sr = 48000):
        """
        input: (L, M) - L - sound size, M - number of sounds
        audio = [sound1, sound2, ...]
        """
        return np.array(list(map(lambda x: camns_audio(sr=sr).set_sounds(x), audio.T)))