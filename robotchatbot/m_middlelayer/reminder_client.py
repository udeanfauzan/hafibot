import uuid
import json
import re
import os
import requests

from datetime import datetime, timedelta
from dateutil.parser import isoparse
from dateutil.relativedelta import relativedelta

from m_middlelayer.middlelayer import real_time


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
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Dictionary of prayer times
        prayer_times = {
            "subuh": data['data']['timings']['Fajr'],
            "dzuhur": data['data']['timings']['Dhuhr'],
            "ashar": data['data']['timings']['Asr'],
            "magrib": data['data']['timings']['Maghrib'],
            "isya": data['data']['timings']['Isha']
        }
    
        # If no match is found, return None
        return  prayer_times
    
    def _preload_reminders(self):
        prayer_times = self.get_prayer_times(city="Bandung", country="Indonesia", method=2)

        today = isoparse(real_time()["datetime"])

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
    #                    MAIN FUNCTION
    # ======================================================
    def check(self):
        self.reminders = self._load()

        now = isoparse(real_time()["datetime"])

        # The list check loop
        for reminder in self.reminders.copy():
            reminder_id = reminder["uuid"]
            reminder_dt = isoparse(reminder["startdate"])
            repeat_type = reminder["repeat_type"]
            called_upon = reminder["called_upon"]

            # Check the time difference
            time_delta = now - reminder_dt

            # Check if reminder had been called
            if called_upon:
                
                if repeat_type is None:
                    # If this reminder is really old
                    if time_delta > timedelta(days=7):
                        self.reminders.remove(reminder)
                        self._save()
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
            if time_delta <= timedelta(minutes=15):
                return True, reminder

            # If it's been more than 15 minutes and the reminder type is repeat
            if time_delta > timedelta(minutes=15) and repeat_type is not None:
                reminder_dt += relativedelta(days=repeat_type)

                reminder["startdate"] = reminder_dt.isoformat()
                self._save()
            
        return False, None

    # ======================================================
    #                  SECONDARY FUNCTION
    # ======================================================
    def update_reminder(self,updated_reminder):
        self.reminders = self._load()

        reminder_id = updated_reminder["uuid"]
        # Find the ID
        for index,reminder in enumerate(self.reminders):
            if reminder["uuid"] == reminder_id:
                self.reminders[index] = updated_reminder
                self._save()
                return

        reminder_task = updated_reminder["task"]
        # Find the task if the ID mismatched
        for index,reminder in enumerate(self.reminders):
            if reminder["task"] == reminder_task:
                self.reminders[index] = updated_reminder
                self._save()
                return
        # Append if it doesn't find either, so a new reminder
        self.reminders.append(updated_reminder)
        self._save()

    def remove_reminder(self,uuid):
        self.reminders = self._load()
        self.reminders = [
            reminder for reminder in self.reminders if reminder["uuid"] != uuid
        ]
        self._save()