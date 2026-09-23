
import requests
from datetime import datetime
import pytz
from dateparser import parse
from dateparser.search import search_dates

# News API
def get_news(entry=1):
    # URL API NEWS
    url = f"https://berita-indo-api-next.vercel.app/api/tempo-news/tekno"
    try:
            # Mengirim permintaan ke API
        response = requests.get(url)
        response.raise_for_status()  # Memastikan permintaan berhasil

            # Mengambil data dalam format JSON
        data = response.json()
        #news = data['data'][1]['title']
        news = data["data"][entry]
        news_data = {
            "title" : news["title"],
            "date"  : news["isoDate"]
        }

        return news_data
    except requests.exceptions.RequestException as e:
        news_data = {
            "error" : "Terjadi kesalahan saat mengambil data berita"
        }

# Time API
def real_time(timezone="Asia/Jakarta"):
    time = datetime.now(pytz.timezone(timezone))

    # Indonesian day and month names
    days = [
        "Senin", "Selasa", "Rabu", "Kamis",
        "Jumat", "Sabtu", "Minggu"
    ]

    months = [
        "Januari", "Februari", "Maret", "April",
        "Mei", "Juni", "Juli", "Agustus",
        "September", "Oktober", "November", "Desember"
    ]

    # UTC offset, e.g. UTC+07:00
    offset = time.strftime("%z")
    timezone_display = f"UTC{offset[:3]}:{offset[3:]}"

    time_now = {
        "date": time.strftime("%Y-%m-%d"),
        "day": days[time.weekday()],
        "month": months[time.month - 1],
        "year": time.strftime("%Y"),
        "time": time.strftime("%H:%M:%S"),
        "time_12h": time.strftime("%I:%M:%S %p"),
        "time_24h": time.strftime("%H:%M:%S"),
        "datetime": time.isoformat(timespec="seconds"),
        "timezone": timezone_display
    }

    return time_now

# Weather API
def get_weather(adm_code="32.73.02.1005"):
    
    # URL API BMKG untuk prakiraan cuaca
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm_code}"

    try:
        # Mengirim permintaan ke API
        response = requests.get(url)
        response.raise_for_status()  # Memastikan permintaan berhasil

        # Mengambil data dalam format JSON
        data = response.json()
        cuaca = data['data'][0]['cuaca'][0][0]['weather_desc']
        suhu = data['data'][0]['cuaca'][0][0]['t']
        kelembapan = data['data'][0]['cuaca'][0][0]['hu']

        # Menyusun hasil dalam dictionary
        weather_info = {
            "cuaca": cuaca,
            "suhu": suhu,
            "kelembapan": kelembapan
        }

        return weather_info
    except requests.exceptions.RequestException as e:
        weather_info = {
            "error" : "Terjadi kesalahan saat mengambil data cuaca"
        }
        return weather_info

# List of predetermined responses (Command is for functions that doesnt really need a qwen response, TBA is well, TBA. Reminder shall go to command check too once its done)
override_check = {
    "weather" : "Maaf, saya belum dilatih untuk cuaca"
}

command_check = {
    "mulai" : "Baik, saya akan kembali bergerak",
    "berhenti" : "Baik, saya akan berhenti bergerak"
}


# Building the dictionary sample for analysis and for Qwen
def context_builder(text, bert_inference, prev_resp=None):
    # Check Moment (True if the bert inference is in the check, else false)
    command_override = bert_inference["intent"] in command_check
    TBA_override = None

    # Full sample for analysis
    fulltext = {
        "input" : text,
        "previous_response" : prev_resp,

        "intent" : bert_inference["intent"],
        "sentiment" : bert_inference["sentiment"],
        "function" : bert_inference["function"],

        "time" : real_time()["datetime"],
        "news" : get_news(1)["title"] if bert_inference["function"] == "news" else None,
        "weather" : get_weather() if bert_inference["function"] == "weather" else None,

        "command_override" : command_override,
        "command_text" : command_check[bert_inference["intent"]] if command_override else None,

        "TBA_override" : None,
        "TBA_text" : override_check[bert_inference["function"]] if TBA_override else None,

        "Reminder_Add" : None,
        "Reminder_Query" : None
    }

    # Partial sample for Qwen response
    qwentext = {
        "input" : text,
        "context" : {
            "intent" : bert_inference["intent"],
            "sentiment" : bert_inference["sentiment"],
        }
    }
    if bert_inference["function"] == "time":
        qwentext["context"]["time"] = {
            "datetime" : real_time()["datetime"]
        }

    if bert_inference["function"] == "news":
        qwentext["context"]["news"] = {
            "title" : get_news(1)["title"]
        }
        
    if bert_inference["function"] == "weather":
        qwentext["context"]["weather"] = get_weather
    return fulltext,qwentext