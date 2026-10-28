
import requests
from datetime import datetime
import pytz
from dateparser import parse
from dateparser.search import search_dates

# News API
def get_news(entry=0):
    url = f"https://berita-indo-api-next.vercel.app/api/tempo-news/tekno"
    try:
        response = requests.get(url, timeout=3.0)
        response.raise_for_status()
        data = response.json()
        articles = data.get("data", [])
        if articles and len(articles) > entry:
            news = articles[entry]
            return {
                "title" : news.get("title", ""),
                "date"  : news.get("isoDate", "")
            }
    except Exception:
        pass
    return {
        "title" : "Perkembangan teknologi AI dan robotika di Indonesia terus berkembang pesat.",
        "date"  : datetime.now().isoformat()
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
    url = f"https://api.bmkg.go.id/publik/prakiraan-cuaca?adm4={adm_code}"
    try:
        response = requests.get(url, timeout=3.0)
        response.raise_for_status()
        data = response.json()
        cuaca = data['data'][0]['cuaca'][0][0]['weather_desc']
        suhu = data['data'][0]['cuaca'][0][0]['t']
        kelembapan = data['data'][0]['cuaca'][0][0]['hu']
        return {
            "cuaca": cuaca,
            "suhu": suhu,
            "kelembapan": kelembapan
        }
    except Exception:
        return {
            "cuaca": "Cerah Berawan",
            "suhu": "26",
            "kelembapan": "75"
        }

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
    text_str = str(text).lower() if text is not None else ""
    command_override = bert_inference.get("intent") in command_check
    TBA_override = False

    is_time = bert_inference.get("function") == "time" or bert_inference.get("intent") == "nanya_waktu" or any(w in text_str for w in ["jam berapa", "pukul berapa", "waktu"])
    is_weather = bert_inference.get("function") == "weather" or bert_inference.get("intent") == "nanya_cuaca" or any(w in text_str for w in ["cuaca", "hujan", "suhu"])
    is_news = bert_inference.get("function") == "news" or bert_inference.get("intent") == "nanya_berita" or any(w in text_str for w in ["berita", "kabar terbaru"])

    rt = real_time()
    weather_data = get_weather() if is_weather else None
    news_data = get_news(0) if is_news else None

    # Full sample for analysis
    fulltext = {
        "input" : text,
        "previous_response" : prev_resp,
        "intent" : bert_inference.get("intent"),
        "sentiment" : bert_inference.get("sentiment"),
        "function" : bert_inference.get("function"),
        "time" : rt["datetime"],
        "news" : news_data["title"] if news_data else None,
        "weather" : weather_data,
        "command_override" : command_override,
        "command_text" : command_check.get(bert_inference.get("intent")) if command_override else None,
        "TBA_override" : False,
        "TBA_text" : None,
        "Reminder_Add" : None,
        "Reminder_Query" : None
    }

    # Partial sample for Qwen response
    qwentext = {
        "input" : text,
        "context" : {
            "intent" : bert_inference.get("intent"),
            "sentiment" : bert_inference.get("sentiment"),
        }
    }
    if is_time:
        qwentext["context"]["time"] = {
            "datetime" : rt["datetime"],
            "display" : f"{rt['day']}, {rt['date']} jam {rt['time'][:5]} WIB"
        }
    if is_news and news_data:
        qwentext["context"]["news"] = {
            "title" : news_data["title"]
        }
    if is_weather and weather_data:
        qwentext["context"]["weather"] = weather_data

    return fulltext, qwentext
