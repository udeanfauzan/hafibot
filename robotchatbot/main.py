from m_distilbert.inference import DistilBertProcessor 
from m_middlelayer import middlelayer as midlayer
from m_qwen.response import QwenResponse 
import json
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import os
from m_voice.voice import AutoSpeech, AutoTTS
from m_middlelayer.reminder import ReminderAdd, ReminderCheck, ReminderCall

def main():
    # Start the models
    dbert = DistilBertProcessor()
    print("DistilBERT Loaded, Qwen Next")
    qwen = QwenResponse()
    print("Qwen Loaded, Have Fun")
    speech = AutoSpeech()
    print("AutoSpeech Loaded, speaking is allowed.")
    text_speech = AutoTTS()
    print("AutoTTS Loaded, robot may speak")
    r_add = ReminderAdd()
    print("Reminder added/loaded. This puny robot can remember a little")
    r_check = ReminderCheck(r_add.json_path)
    print("Reminder can be queried now.")
    r_call = ReminderCall(r_add.json_path)
    print("Reminder can be called now. I'm sorry in advance for the inconvenience.")

    # Setting an empty previous response
    prev_resp = None

    # Initializing empty row for documentation
    result = []

    # Chatbot Loop
    while True:
        # Start with checking the reminders and prepare alarm if needed
        reminder_check, reminder_id = r_check.check()
        if reminder_check:
            r_call.call(reminder_id=reminder_id, output_fn=text_speech.speak)
            continue

        # Input Req (Change this to a voice based one later)
        # text = input("Teks : ") # Text Version
        text = speech.listen_text()
        text = text.strip()

        # Cancelling empty phrase
        if text == "":
            continue
        
        # (For testing purpose) Early stop
        if text.lower().rstrip() == "susu manis":
            print("Robot selesai...")
            break

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
        elif fulltext["function"] == "reminder":
            reminder = r_add.add(
                text_ori = fulltext["input"],
                input_fn = speech.listen_text,
                output_fn = text_speech.speak
            )
            if reminder is not None:
                response = f"Baik, pengingat {reminder['task']} telah disimpan."
                fulltext["Reminder_Add"] = reminder
            else:
                response = "Pengingat tidak berhasil disimpan"
                fulltext["Reminder_Add"] = "Failed"
        elif fulltext["function"] == "agenda_query":
            responses, query_res = r_check.query(fulltext["input"])
            response = responses
            fulltext["Reminder_Query"] = query_res
        elif fulltext["function"] == "repeat":
            response = prev_resp
        else:
            response = qwen.generate_response(qwentext)

        # Fill the response to the previous response and fulltext
        prev_resp = response if not fulltext["command_override"] and not fulltext["TBA_override"] else prev_resp
        fulltext["response"] = response

        # Return the result
        print("Response : ",response)
        text_speech.speak(response)

        # Append to the row
        result.append(fulltext)

    # Retrieve
    return result


if __name__ == "__main__":
    time = midlayer.real_time()
    timeoftest = f"{time["date"]}_{time['time'].replace(':', '-')}"
    testresult = main()
    os.makedirs("robotchatbot/m_testresult", exist_ok=True)
    with open(os.path.join("robotchatbot/m_testresult",f"{timeoftest}-TestResult.json"),"w") as f:
        json.dump(testresult,f,indent=4)