import io
import re
import wave
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False


class AutoSpeech:
    """Google Indonesian Speech Recognition (id-ID)."""

    def __init__(self, model_dir=None):
        self.recognizer = sr.Recognizer()
        self.samplerate = 16000
        self.channels = 1

    def record(self, timeout=None, phrase_time_limit=10):
        with sr.Microphone(sample_rate=self.samplerate) as source:
            print('Menyesuaikan dengan lingkungan...')
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print('Mendengarkan...')
            audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        print('Berhasil direkam.')
        return audio

    def parse_number(self, words):
        ones = {
            'nol': 0, 'satu': 1, 'dua': 2, 'tiga': 3, 'empat': 4,
            'lima': 5, 'enam': 6, 'tujuh': 7, 'delapan': 8, 'sembilan': 9
        }
        total = 0
        current = 0
        i = 0
        while i < len(words):
            word = words[i]
            if word in ones:
                current += ones[word]
            elif word == 'sepuluh':
                current += 10
            elif word == 'sebelas':
                current += 11
            elif word == 'seratus':
                total += 100
                current = 0
            elif word == 'seribu':
                total += 1000
                current = 0
            elif word == 'belas':
                if current == 0:
                    current = 10
                else:
                    current += 10
            elif word == 'puluh':
                if current == 0:
                    current = 1
                current *= 10
            elif word == 'ratus':
                if current == 0:
                    current = 1
                current *= 100
            elif word == 'ribu':
                if current == 0:
                    current = 1
                total += current * 1000
                current = 0
            elif word == 'juta':
                if current == 0:
                    current = 1
                total += current * 1_000_000
                current = 0
            i += 1
        return total + current

    def normalize_numbers(self, text):
        number_words = {
            'nol', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan',
            'sepuluh', 'sebelas', 'seratus', 'seribu', 'belas', 'puluh', 'ratus', 'ribu', 'juta'
        }
        words = text.lower().split()
        result = []
        number_buffer = []

        def flush_number():
            if number_buffer:
                number = self.parse_number(number_buffer)
                result.append(str(number))
                number_buffer.clear()

        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word in number_words:
                number_buffer.append(clean_word)
            else:
                flush_number()
                result.append(word)

        flush_number()
        return ' '.join(result)

    def audio_to_text(self, audio, samplerate=16000):
        # Google Speech Recognition (id-ID)
        try:
            if isinstance(audio, sr.AudioData):
                text = self.recognizer.recognize_google(audio, language='id-ID').strip()
            elif isinstance(audio, bytes):
                bio = io.BytesIO(audio)
                with sr.AudioFile(bio) as source:
                    audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language='id-ID').strip()
            elif isinstance(audio, np.ndarray):
                int_samples = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                bio = io.BytesIO()
                with wave.open(bio, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(samplerate)
                    wf.writeframes(int_samples.tobytes())
                bio.seek(0)
                with sr.AudioFile(bio) as source:
                    audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language='id-ID').strip()
            else:
                text = ''

            if text:
                text = self.normalize_numbers(text)
                print('Saya mendengar (Google ID):', text)
                return text
        except sr.UnknownValueError:
            print('Google ASR: Suara tidak terdengar jelas.')
            return ''
        except Exception as e:
            print(f'Google ASR error: {e}')
            return ''


class AutoTTS:
    def __init__(self):
        pass

    def speak(self, text):
        if not text:
            return
        tts = gTTS(text=text, lang='id')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio = AudioSegment.from_file(audio_buffer, format='mp3')
        samples = audio.get_array_of_samples()
        if audio.channels > 1:
            samples = np.array(samples).reshape((-1, audio.channels))
        if HAS_SD:
            sd.play(samples, samplerate=audio.frame_rate)
            sd.wait()


if __name__ == '__main__':
    speech = AutoSpeech()
    tts = AutoTTS()
    tts.speak('Tes Google ID ASR')
