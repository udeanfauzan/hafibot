import speech_recognition as sr
from gtts import gTTS
import io
from pydub import AudioSegment
import numpy as np
import re

class AutoSpeech:
    def __init__(self):
        self.recognizer = sr.Recognizer()

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

    def audio_to_text(self, audio_data):
        audio_buffer = io.BytesIO(audio_data)
        with sr.AudioFile(audio_buffer) as source:
            input_audio = self.recognizer.record(source)
        try:
            text = self.recognizer.recognize_google(input_audio, language='id-ID')
            text = self.normalize_numbers(text)
            print(f'Saya mendengar (Google ID): {text}')
            return text
        except sr.UnknownValueError:
            return ''
        except Exception as e:
            print(f'Google ASR Error: {e}')
            return ''

class AutoTTS:
    def __init__(self):
        pass

    def text_to_audio(self, text):
        tts = gTTS(text=text, lang='id')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio = AudioSegment.from_file(audio_buffer, format='mp3')
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format='wav')
        wav_buffer.seek(0)
        return wav_buffer.read()
