import io
import sounddevice as sd
import soundfile as sf

class AutoSpeech:
    def __init__(self,samplerate = 16000, channels = 1):
        self.samplerate = samplerate
        self.channels = channels

    def record(self,duration=5):
        print("Mendengarkan...")
        audio = sd.rec(int(duration*self.samplerate),samplerate=self.samplerate, channels=self.channels,dtype="float32")

        sd.wait()

        print("Rekaman selesai.")

        wav_buffer = io.BytesIO()

        sf.write(wav_buffer,audio,samplerate=self.samplerate,format="WAV")

        wav_buffer.seek(0)

        return wav_buffer.read()

class AutoTTS:
    def __init__(self):
        pass

    def play(self,audio_data):
        wav_buffer = io.BytesIO(audio_data)
        audio,samplerate = sf.read(wav_buffer,dtype="float32")

        print("Memutar respon...")

        sd.play(audio,samplerate)
        sd.wait()

        print("Audio selesai diputar")

if __name__ == "__main__":
    speech = AutoSpeech()

    audio_data = speech.record(duration=5)

    with open("recorded.wav","wb") as file:
        file.write(audio_data)

    print("Recording saved.")