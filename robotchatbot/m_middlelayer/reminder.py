import uuid
import json
import re
import os
import requests

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from c_dateparser import reminder_time_normalize, reminder_followup
from middlelayer import real_time

class ReminderAdd:
    # ======================================================
    #                   INITIALIZATION
    # ======================================================
    def __init__(self,json_path=None):
        # Checking the reminder.json
        if json_path is None:
            json_path = os.path.join(os.path.dirname(__file__),"reminder.json")

        self.json_path = json_path

        # Make a json file if it exists
        if not os.path.exists(self.json_path):
            self.reminders = []
            self._preload_reminders()
            self._save()
        else:
            self.reminders = self._load()

    # ======================================================
    #                 JSON LOAD AND SAVE
    # ======================================================
    def _load(self):
        try:
            with open(self.json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data,list):
                return []

            return data
        except (json.JSONDecodeError,OSError):
            return []

    def _save(self):
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(self.reminders, file, ensure_ascii=False, indent=4)

    # ======================================================
    #                 PRELOADED REMINDERS
    # ======================================================
    def get_prayer_times(self, city, country, method):
        url = f"http://api.aladhan.com/v1/timingsByCity?city={city}&country={country}&method={method}"
        try:
            response = requests.get(url, timeout=2.0)
            response.raise_for_status()
            data = response.json()
            prayer_times = {
                "subuh": data['data']['timings']['Fajr'],
                "dzuhur": data['data']['timings']['Dhuhr'],
                "ashar": data['data']['timings']['Asr'],
                "magrib": data['data']['timings']['Maghrib'],
                "isya": data['data']['timings']['Isha']
            }
            return prayer_times
        except Exception:
            return {
                "subuh": "04:45",
                "dzuhur": "12:00",
                "ashar": "15:15",
                "magrib": "18:05",
                "isya": "19:15"
            }
    
    def _preload_reminders(self):
        prayer_times = self.get_prayer_times(city="Bandung", country="Indonesia", method=2)

        today = datetime.fromisoformat(real_time()["datetime"])

        for prayer,prayer_time in prayer_times.items():
            hour,minute = map(int, prayer_time.split(":"))
            startdate = today.replace(hour = hour, minute = minute, second=0, microsecond=0).isoformat()

            reminder = {
                "uuid" : str(uuid.uuid4()),
                "task" : f"waktu solat {prayer}",
                "startdate" : startdate,
                "repeat_type" : 1,
                "called_upon" : False
            }

            self.reminders.append(reminder)

    # ======================================================
    #                    TEXT STRIPPING
    # ======================================================
    def strip_text(self,text):
        text = text.lower()

        reminder_key = ["ingatkan", "ingetin", "jangan lupa", "catat", "kasih tau", "ingatin", "ingetin dong"]

        for keyword in reminder_key:
            text = re.sub(rf"\b{re.escape(keyword)}\b","",text,count=1)

        text = re.sub(r"\s+", " ",text).strip()

        text = re.sub(r"^[,\s]+|[,\s]+$","",text)

        return text

    
    # ======================================================
    #                    CONFIRMATIONS
    # ======================================================
    def confirmation(self,task,result,repeat_type,input_fn=input,output_fn=print):

        periods = { 
            "subuh" : range(3,6),
            "pagi" : range(1,12),
            "siang" : range(10,16),
            "sore" : range(14,19),
            "maghrib" : range(18,19),
            "malam" : list(range(19,24)) + list(range(0,4))
        }

        months = {
            1: "januari",
            2: "februari",
            3: "maret",
            4: "april",
            5: "mei",
            6: "juni",
            7: "juli",
            8: "agustus",
            9: "september",
            10: "oktober",
            11: "november",
            12: "desember"
        }

        dt = datetime.fromisoformat(result)

        hour = dt.hour
        minute = dt.minute

        if hour in periods["maghrib"]:
            period = "maghrib"
        elif hour in periods["malam"]:
            period = "malam"
        elif hour in periods["subuh"]:
            period = "subuh"
        elif hour in periods["siang"]:
            period = "siang"
        elif hour in periods["sore"]:
            period = "sore"
        elif hour in periods["pagi"]:
            period = "pagi"
        else:
            period = "pagi"

        if hour == 0:
            hour = 12
        elif hour > 12:
            hour -= 12
        else:
            hour = hour

        if minute:
            display_hour = f"{period} jam {hour} lewat {minute} menit"
        else:
            display_hour = f"{period} jam {hour}"

        month_name = months[dt.month]

        if repeat_type is None:
            message = f"Pengingat untuk {task} mulai {dt.day} {month_name} {dt.year}, {display_hour}. Apakah benar?"
        else:
            message = f"Pengingat untuk {task} mulai {dt.day} {month_name} {dt.year}, {display_hour}. Pengingat ini diulang setiap {repeat_type} hari. Apakah benar?"

        output_fn(message)

        confirm = input_fn().strip().lower()

        yes_check = ["ya","iya","iyap","betul","benar","benar sekali","betul sekali","sudah","oke","ok","iya benar", "iyo"]
        no_check = ["tidak","nggak","gak","ga","enggak","salah","belum","bukan", "ganti"]

        if any(word == confirm for word in yes_check):
            return True
        elif any(word == confirm for word in no_check):
            return False
        else:
            return False
        
    # ======================================================
    #                FOLLOW UP FUNCTION
    # ======================================================
    def _followup(self, reminder, input_fn=input, output_fn=print):
        followup_prompt = {
            "bulan" : "Bulan kapan?",
            "date" : "tanggal berapa pengingatnya?",
            "time" : "jam berapa pengingatnya?"
        }
        while (reminder["weirddate"] or reminder["relative"] or reminder["missing"] is not None):
            if reminder["weirddate"]:
                prompt = "Bisa diulangi lagi waktunya?"
            else:
                prompt = followup_prompt.get(reminder["missing"],"Bisa diulangi lagi waktunya?")

            output_fn(prompt)
            followup_input = input_fn()
            reminder,completion = reminder_followup(followup_input, reminder)
        
            if completion: break

        return reminder

    # ======================================================
    #                REMINDER ADD FUNCTION
    # ======================================================
    def add(self,text_ori, input_fn=input, output_fn=print):
        text = text_ori.strip().lower()

        # Step 1 : Parse the date time
        reminder_parsed = reminder_time_normalize(text)

        # Step 2 : Followup to check missing data (add messages to output later)
        reminder_parsed = self._followup(reminder_parsed, input_fn, output_fn)

        # Step 3 : Strip the input word of the reminder keyword
        task = self.strip_text(text_ori)

        # Step 4 : Ask for confirmation. If False, loop back to step 2
        confirmed = self.confirmation(task, reminder_parsed["result"],reminder_parsed["repeat_type"], input_fn, output_fn)
        while not confirmed:
            reminder_parsed["weirddate"] = True
            reminder_parsed = self._followup(reminder_parsed, input_fn, output_fn)
            confirmed = self.confirmation(task, reminder_parsed["result"],reminder_parsed["repeat_type"], input_fn, output_fn)

        # Step 5 : Append and save
        new_reminder = {
            "uuid" : str(uuid.uuid4()),
            "task" : task,
            "startdate" : reminder_parsed["result"],
            "repeat_type" : reminder_parsed["repeat_type"],
            "called_upon" : False
        }
        self.reminders.append(new_reminder)
        self._save()

        return new_reminder

class ReminderCheck:
    # ======================================================
    #                   INITIALIZATION
    # ======================================================
    def __init__(self,json_path=None):
        if json_path is None:
            json_path = os.path.join(os.path.dirname(__file__),"reminder.json")

        self.json_path = json_path

        if not os.path.exists(self.json_path):
            self.reminders = []
        else:
            self.reminders = self._load()

    # ======================================================
    #                 JSON LOAD AND SAVE
    # ======================================================
    def _load(self):
        try:
            with open(self.json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data,list):
                return []

            return data
        except (json.JSONDecodeError,OSError):
            return []

    def _save(self):
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(self.reminders, file, ensure_ascii=False, indent=4)

    # ======================================================
    #                    MAIN FUNCTION
    # ======================================================
    def check(self):
        self.reminders = self._load()

        now = datetime.fromisoformat(real_time()["datetime"])
        now_naive = now.replace(tzinfo=None) if now.tzinfo is not None else now

        # The list check loop
        for reminder in self.reminders.copy():
            reminder_id = reminder["uuid"]
            try:
                reminder_dt = datetime.fromisoformat(reminder["startdate"])
            except Exception:
                continue
            rem_naive = reminder_dt.replace(tzinfo=None) if reminder_dt.tzinfo is not None else reminder_dt
            repeat_type = reminder["repeat_type"]
            called_upon = reminder["called_upon"]

            # Check the time difference safely
            time_delta = now_naive - rem_naive

            # Check if reminder had been called
            if called_upon:
                
                if repeat_type is None:
                    # If this reminder is really old
                    if time_delta > timedelta(days=7):
                        self.reminders.remove(reminder)
                        self._save()
                        continue

                    # If its not that old yet
                    else:
                        continue

                # Modify the startdate
                reminder_dt += relativedelta(days=repeat_type)

                # Modify the reminder row
                reminder["startdate"] = reminder_dt.isoformat()
                reminder["called_upon"] = False

                self._save()

                continue

            # If it's not the time yet
            if time_delta.total_seconds() < 0:
                continue

            # If it's at most 10 minutes since the time
            if time_delta <= timedelta(minutes=10):
                return True, reminder_id

            # If it's been more than 10 minutes and the reminder type is repeat
            if time_delta > timedelta(minutes=10) and repeat_type is not None:
                reminder_dt += relativedelta(days=repeat_type)

                reminder["startdate"] = reminder_dt.isoformat()
                self._save()
            
        return False, None

    # ======================================================
    #                  SECONDARY FUNCTION
    # ======================================================
    def query(self,text_ori):
        self.reminders = self._load()

        ignored_words = {
            "apa",
            "ada",
            "aku",
            "saya",
            "yang",
            "untuk",
            "dengan",
            "dari",
            "ke",
            "di",
            "dan",
            "atau",
            "kapan",
            "dimana",
            "mana",
            "jadwal",
            "reminder",
            "pengingat",
            "ingat",
            "ingatkan",
            "tolong",
            "dong",
            "ya",
            "nggak",
            "gak",
            "ga"
        }

        # Split and filter text
        text = text_ori.strip().lower()

        query_text = set(re.findall(r"\b\w+\b",text))

        query_text -= ignored_words

        query_res = []

        # Checking each row
        for reminder in self.reminders:
            task_words = set(re.findall(r"\b\w+\b",reminder["task"]))

            if query_text & task_words:
                query_res.append({
                    "task" : reminder["task"],
                    "date" : reminder["startdate"]
                })

        # If no specific agenda is found, return all
        if not query_res:
            query_res = [{
                "task" : reminder["task"],
                "date" : reminder["startdate"]
            } for reminder in self.reminders]

        # Yet another:tm: variable dictionary
        periods = { 
            "subuh" : range(3,6),
            "pagi" : range(1,12),
            "siang" : range(10,16),
            "sore" : range(14,19),
            "maghrib" : range(18,19),
            "malam" : list(range(19,24)) + list(range(0,4))
        }

        months = {
            1: "januari",
            2: "februari",
            3: "maret",
            4: "april",
            5: "mei",
            6: "juni",
            7: "juli",
            8: "agustus",
            9: "september",
            10: "oktober",
            11: "november",
            12: "desember"
        }

        messages = []
        for reminder in query_res:
            reminder_dt = datetime.fromisoformat(reminder["date"])

            hour = reminder_dt.hour
            minute = reminder_dt.minute

            if hour in periods["maghrib"]:
                period = "maghrib"
            elif hour in periods["malam"]:
                period = "malam"
            elif hour in periods["subuh"]:
                period = "subuh"
            elif hour in periods["siang"]:
                period = "siang"
            elif hour in periods["sore"]:
                period = "sore"
            elif hour in periods["pagi"]:
                period = "pagi"
            else:
                period = "pagi"

            if hour == 0:
                hour = 12
            elif hour > 12:
                hour -= 12
            else:
                hour = hour

            if minute:
                display_hour = f"{period} jam {hour} lewat {minute} menit"
            else:
                display_hour = f"{period} jam {hour}"

            month_name = months[reminder_dt.month]

            messages.append(f"{reminder['task']} pada {reminder_dt.day} {month_name} {reminder_dt.year}, {display_hour}")

        response = f"Anda ada pengingat untuk {messages[0]}."
        for message in messages[1:]:
            response += f" Dan juga ada {message}"

        if not query_text:
            response = f"Tidak ada pengingat yang sesuai. Tetapi " + response
        return response, query_res

class ReminderCall:
    # ======================================================
    #                   INITIALIZATION
    # ======================================================
    def __init__(self,json_path=None):
        if json_path is None:
            json_path = os.path.join(os.path.dirname(__file__),"reminder.json")

        self.json_path = json_path

        if not os.path.exists(self.json_path):
            self.reminders = []
        else:
            self.reminders = self._load()

    # ======================================================
    #                 JSON LOAD AND SAVE
    # ======================================================
    def _load(self):
        try:
            with open(self.json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            if not isinstance(data,list):
                return []

            return data
        except (json.JSONDecodeError,OSError):
            return []

    def _save(self):
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(self.reminders, file, ensure_ascii=False, indent=4)

    # ======================================================
    #                  MAIN FUNCTION
    # ======================================================
    def call(self,reminder_id,output_fn=print):
        self.reminders = self._load()

        # Step 1 : Locate the reminder row
        for reminder in self.reminders:
            if reminder["uuid"] != reminder_id:
                continue

            task = reminder["task"]
            dt = datetime.fromisoformat(reminder["startdate"])
            hour = dt.hour
            minute = dt.minute

            periods = {
                "subuh" : range(3,6),
                "pagi" : range(1,12),
                "siang" : range(10,16),
                "sore" : range(14,19),
                "maghrib" : range(18,19),
                "malam" : list(range(19,24)) + list(range(0,4))
            }

            if hour in periods["maghrib"]:
                period = "maghrib"
            elif hour in periods["malam"]:
                period = "malam"
            elif hour in periods["subuh"]:
                period = "subuh"
            elif hour in periods["siang"]:
                period = "siang"
            elif hour in periods["sore"]:
                period = "sore"
            elif hour in periods["pagi"]:
                period = "pagi"
            else:
                period = "pagi"

            if hour == 0:
                hour = 12
            elif hour > 12:
                hour -= 12
            else:
                hour = hour

            if minute:
                display_hour = f"{period} jam {hour} lewat {minute} menit"
            else:
                display_hour = f"{period} jam {hour}"

            message = f"Pengingat! Kamu ada pengingat! {task}!!! Pada {display_hour}"

            output_fn(message)

            reminder["called_upon"] = True

            self._save()

        return False