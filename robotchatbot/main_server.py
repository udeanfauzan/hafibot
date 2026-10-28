from m_distilbert.inference import DistilBertProcessor 
from m_middlelayer import middlelayer as midlayer
from m_qwen.response import QwenResponse 
import json
import socket
import numpy as np
import base64
from datetime import datetime
import os
from m_voice.voice_server import AutoSpeech, AutoTTS
from m_middlelayer.reminder_server import ReminderAdd, ReminderCheck, ReminderCall

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
    # Start the models
    dbert = DistilBertProcessor()
    print("DistilBERT Loaded, Qwen Next")
    qwen = QwenResponse()
    print("Qwen Loaded, Have Fun")
    speech = AutoSpeech()
    print("AutoSpeech Loaded, speaking is allowed.")
    tts = AutoTTS()
    print("AutoTTS Loaded, robot may speak")
    r_add = ReminderAdd()
    print("Reminder added/loaded. This puny robot can remember a little")
    r_check = ReminderCheck(r_add.json_path)
    print("Reminder can be queried now.")
    r_call = ReminderCall(r_add.json_path)
    print("Reminder can be called now. I'm sorry in advance for the inconvenience.")

    # Setting an empty previous response
    prev_resp = None
    result = []

    # Server init
    HOST = "0.0.0.0"
    PORT = 5000

    server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)

    server.bind((HOST,PORT))
    server.listen(1)
    print("Waiting for Jetson...")

    client,address = server.accept()

    print(f"Jetson Connected, {address}")

    # Server Callbacks
    def send_text(text):
        send_message(client,{
            "type" : "text",
            "text" : text
        })

    def receive_text():
        message = receive_message(client)

        if message is None: return ""
        if message.get("type") == "text":
            return message.get("text","")

        return ""

    def send_audio(audio_bytes):
        send_message(client,{
            "type" : "audio",
            "data" : encode_audio(audio_bytes)
        })

    def reminder_output(text,end=False):
        send_text(text)

        audio = tts.text_to_audio(text)
        send_audio(audio)

        send_message(client,{
            "type" : "reminder_end",
            "content" : end
        })

    def reminder_input():
        message = receive_message(client)

        if message is None: return ""
        if message.get("type") == "text":
            return message.get("text","")
        if message.get("type") == "audio":
            audio_bytes = decode_audio(message["data"])
            text = speech.audio_to_text(audio_bytes)
            return text.strip()
        return ""

    # Main Loop Time
    while True:
        message = receive_message(client)

        if message is None:
            print("Jetson Disconnected")
            break

        message_type = message.get("type")

        # If a reminder is happening
        if message_type == "alarm":
            reminder = message["reminder"]
            print("Alarm received.",reminder)

            message, updated_reminder = r_call.call(reminder)

            # Holy Trinity of Sending
            send_text(message)
            wav_bytes = tts.text_to_audio(message)
            send_audio(wav_bytes)
            send_message(client,{
                "type" : "reminder_update",
                "reminder" : updated_reminder
            })
            
            continue

        # Normal input
        audio_bytes = decode_audio(message["data"])
        text = speech.audio_to_text(audio_bytes)
        text = text.strip()
        if text == "":
            continue

        # Labels Inference (Intent, Sentiment, Function)
        labels = dbert.call_inference(text)

        # Context Enriching
        fulltext, qwentext = midlayer.context_builder(
            text = text,
            bert_inference = labels,
            prev_resp = prev_resp
        )

        # Decision Making (check both command_override and TBA_override before deciding Qwen response)
        if fulltext["command_override"]:
            response = fulltext["command_text"]
        elif fulltext["TBA_override"]:
            response = fulltext["TBA_text"]
        elif fulltext["function"] == "repeat":
            response = prev_resp
        elif fulltext["function"] == "reminder":
            send_message(client,{
                "type" : "reminder_start",
                "data" : True
            })

            reminder = r_add.add(
                text_ori=fulltext["input"],
                input_fn=reminder_input,
                output_fn=reminder_output
            )

            send_message(client,{
                "type" : "updated_reminder",
                "reminder" : reminder
            })
            continue
        elif fulltext["function"] == "agenda_query":
            responses, query_res = r_check.query(fulltext["input"])
            response = responses
            fulltext["Reminder_Query"] = query_res
        else:
            response = qwen.generate_response(qwentext)

        # Fill the response to the previous response and fulltext
        prev_resp = response if not fulltext["command_override"] and not fulltext["TBA_override"] else prev_resp
        fulltext["response"] = response

        # Return the result
        print("Response : ",response)

        # Guess who's back (sending both text and audio, in that order)
        send_text(response)
        wav_bytes = tts.text_to_audio(response)
        send_audio(wav_bytes)

        # Append to the row
        result.append(fulltext)

    # Retrieve
    return result

if __name__ == "__main__":
    time = midlayer.real_time()
    timeoftest = f"{time['date']}_{time['time'].replace(':', '-')}"
    testresult = main()
    os.makedirs("robotchatbot/m_testresult", exist_ok=True)
    with open(os.path.join("robotchatbot/m_testresult",f"{timeoftest}-TestResult.json"),"w") as f:
        json.dump(testresult,f,indent=4)
