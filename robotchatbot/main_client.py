from m_middlelayer import middlelayer as midlayer
import socket
import base64
import json
import os
from m_voice.voice_client import AutoSpeech, AutoTTS
from m_middlelayer.reminder_client import ReminderCheck

# Preliminary functions
def send_message(sock,message):
    data = json.dumps(message,ensure_ascii=False).encode("utf-8")
    size = len(data)
    sock.sendall(size.to_bytes(4,byteorder="big"))
    sock.sendall(data)

def receive_exact(sock,size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size-len(data))
        if not chunk:
            return None
        data += chunk
    return data

def receive_message(sock):
    size_data = receive_exact(sock, 4)
    if not size_data: return None
    size = int.from_bytes(size_data,byteorder="big")

    data = receive_exact(sock, size)
    if not data: return None
    return json.loads(data.decode("utf-8"))

def encode_audio(audio_bytes):
    return base64.b64encode(audio_bytes).decode("ascii")

def decode_audio(audio_string):
    return base64.b64decode(audio_string.encode("ascii"))

def main():
    # Initializing classes
    speech = AutoSpeech()
    print("AutoSpeech Loaded, speaking is allowed.")
    tts = AutoTTS()
    print("AutoTTS Loaded, robot may speak")
    r_check = ReminderCheck()
    print("Reminder Checker starts now.")

    # Server stuff
    SERVER_IP = "100.x.x.x"
    SERVER_PORT = 5000

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("Connecting to server...")
    client.connect((SERVER_IP,SERVER_PORT))

    print("Connected to server!")

    # Jetson Chatbot Loop
    while True:
        reminder_flag, reminder = r_check.check()

        # If a reminder is due
        if reminder_flag:
            print("Reminder found : ",reminder)

            send_message(client,{
                "type" : "alarm",
                "reminder" : reminder
            })

            # The Alarm Loop
            while True:
                message = receive_message(client)

                if message is None:
                    print("Server Disconnected")
                    return

                # Holy Trinity of Receiving
                if message.get("type") == "text":
                    print("Alarm: ",message.get("text",""))

                if message.get("type") == "audio":
                    wav_bytes = decode_audio(message["data"])
                    tts.play(wav_bytes)

                if message.get("type") == "reminder_update":
                    r_check.update_reminder(message["reminder"])
                    break

            continue

        # Listening and sending audio
        audio_bytes = speech.record_audio()
        send_message(client,{
            "text" : "audio",
            "data" : encode_audio(audio_bytes)
        })

        # The Core Loop
        while True:
            message = receive_message(client)

            if message is None:
                print("Server Disconnected")
                return

            # Yet another Receiver
            if message.get("type") == "text":
                print("Server : ",message.get("text",""))
                continue

            if message.get("type") == "audio":
                wav_bytes = decode_audio(message["data"])
                tts.play(wav_bytes)
                break

            if message.get("type") == "reminder_start":
                reminder_add = message.get("data")
                break

        if reminder_add:
            # Began, the Reminder Loop has
            while True:
                message = receive_message(client)
                if message is None:
                    print("Server Disconnected")
                    return

                if message.get("type") == "text":
                    print("Server : ",message.get("text",""))
                    continue

                if message.get("type") == "audio":
                    wav_bytes = decode_audio(message["data"])
                    tts.play(wav_bytes)
                    continue

                if message.get("type") == "reminder_end":
                    if not message.get("content",False):
                        audio_bytes = speech.record_audio()
                        send_message(client,{
                            "text" : "audio",
                            "data" : encode_audio(audio_bytes)
                        })
                        continue

                    if message.get("content",False):
                        continue

                if message.get("type") == "updated_reminder":
                    r_check.update_reminder(message["reminder"])
                    break

    client.close()

if __name__ == "__main__":
    main()