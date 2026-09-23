from middlelayer import real_time
import re
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from dateparser.search import search_dates

# ==========================================
#               DICTIONARIES
# ==========================================

relative_normalizetext = { # Regex to detect relative time phrases (e.g. "3 hari lagi" "7 bulan ke depan")
    "lagi" : r"(\d+)\s+(detik|menit|jam|hari|minggu|bulan|tahun)\s+lagi",
    "kedepan" : r"(\d+)\s+(detik|menit|jam|hari|minggu|bulan|tahun)\s+kedepan",
    "ke depan" : r"(\d+)\s+(detik|menit|jam|hari|minggu|bulan|tahun)\s+ke depan",
    "dari sekarang" : r"(\d+)\s+(detik|menit|jam|hari|minggu|bulan|tahun)\s+dari sekarang",
    "di" : r"di\s+(\d+)\s+(detik|menit|jam|hari|minggu|bulan|tahun)\s+",
    "nanti" : r"(\d+)\s+(detik|menit|jam|hari|minggu|bulan|tahun)\s+nanti" or r"nanti\s+(\d+)\s+(detik|menit|jam|hari|minggu|bulan|tahun)",
    "besok" : r"\bbesok\b",
    "bulan depan" : r"\bbulan depan\b",
    "tahun depan" : r"\btahun depan\b"
}

periods = { # Time period associated with hour times
    "subuh" : range(3,6),
    "pagi" : range(1,12),
    "siang" : range(10,16),
    "sore" : range(14,19),
    "maghrib" : range(18,19),
    "malam" : list(range(19,24)) + list(range(0,4))
}

months = { #  Months Num to Name
    1 : "januari", 2 : "februari", 3 : "maret", 4 : "april", 5 : "mei", 6 : "juni", 7 : "juli", 8 : "agustus", 9 : "september", 10 : "oktober", 11 : "november", 12 : "desember"
}

months_name = { # Months Name to Num
    "januari" : 1, "februari" : 2, "maret" : 3, "april" : 4, "mei" : 5, "juni" : 6, "juli" : 7, "agustus" : 8, "september" : 9, "oktober" : 10, "november" : 11, "desember" : 12
}

timeset = { # Use this for search_dates settings=
        "TIMEZONE": "Asia/Jakarta","RETURN_AS_TIMEZONE_AWARE": True
        }

weekdays = { # Weekdays labelling
    "senin" : 0, "selasa" : 1, "rabu" : 2, "kamis" : 3, "jumat" : 4, "sabtu" : 5, "minggu" : 6
}

temporal_keywords = [ # Derived from the dataset, detecting words to replace 
    "tanggal",
    "jam",
    "hari ini",
    "hari",
    "besok",
    "lusa",
    "nanti",
    "sore ini",
    "pagi ini",
    "siang ini",
    "malam ini",
    "nanti pagi",
    "nanti siang",
    "nanti sore",
    "nanti malam",
    "minggu depan",
    "bulan depan",
    "tahun depan",
]
temporal_keywords += list(weekdays.keys())
temporal_keywords += list(months.values())
temporal_keywords.sort(key=len,reverse=True)

fix_period = { # Fixed time period if the time period is not specified with hour time
    "subuh" : 3,
    "pagi" : 8,
    "siang" : 12,
    "zuhur" : 12,
    "dzuhur" : 12,
    "sore" : 15,
    "asar" : 15,
    "ashar" : 15,
    "maghrib" : 18,
    "isya" : 19,
    "malam" : 20,
    "tengah malam" : 0,
}

# ==========================================
#          LOWER LEVEL FUNCTIONS
# ==========================================

def relative_normalize(text): # Normalizing relative time phrases to one searchable by search_dates. Run after repeat_category. If relative is true, ask the user for lower level timemarks
    relative = False
    change = False
    for keyword, phrase in relative_normalizetext.items():
        if keyword in text:
            if keyword == "besok":
                besok_weekday = False
                for day in weekdays:
                    if re.search(rf"\bbesok\s+{day}\b",text):
                        besok_weekday = True
                        break
                if besok_weekday: continue
                else: 
                    textfix = re.sub(phrase, "dalam 1 hari",text)
                    change = True
            elif keyword == "bulan depan":
                textfix = re.sub(phrase, "dalam 1 bulan",text)
                change = True
            elif keyword == "tahun depan":
                textfix = re.sub(phrase, "dalam 1 tahun",text)
                change = True
            else:
                textfix = re.sub(phrase, r"dalam \1 \2", text)
                change = True
    if change:
        relative = True
        return textfix, relative
    else:
        relative = False
        return text,relative

def repeat_flag(text): # Marking the sentence if it wants a repeatable reminder. Run first
    repetitive = False
    if "tiap" in text or "setiap" in text:
        repetitive = True
    return repetitive

def time_normalize(text): # Normalizing various time phrases. Useful to determine the exact hour based on some dubious phrases. Precise to half hours. Also separates the date and time phrases. Run last, before the search_date
    timerange = None
    has_time = True

    date_check = re.search(r"\b\d{4}-\d{2}-\d{2}\s*\b",text)
    if date_check:
        text = re.sub(r"\b\d{4}-\d{2}-\d{2}\s*\b","",text)

    hour_match = re.search(r"\b(\d{1,2})\b", text)
    if not hour_match:
        period_match = None
        hour_time = None
        for period, hourtime in fix_period.items():
            if re.search(rf"\b{re.escape(period)}\b",text):
                period_match = period
                hour_time = hourtime
                break

        if period_match:
            text = re.sub(rf"\b{re.escape(period_match)}\b", f"jam {hour_time}",text,count = 1)
            hour_match = re.search(r"\b(\d{1,2})\b", text)

        else: 
            if date_check:
                text_date = text + " " + date_check.group(0)
            textfix = None
            has_time = False
            return {
                "texttime" : textfix, 
                "textdate" : text_date.strip() if date_check else None, 
                "has_time" : has_time
            }
    
    hour = int(hour_match.group(1))
    minute = 0

    half_before = re.search(r"\bsetengah\s+(\d{1,2})\b",text)
    half_after = re.search(r"\b(\d{1,2})\s+setengah\b",text)
    
    if hour == 0:
        hour = 12

    for word in periods.keys():
        if word in text:
            timerange = word
            break

    if timerange:
        valid_hours = periods[timerange]
        candidates = [hour]

        if hour<12:
            candidates.append(hour+12)

        for candidate in candidates:
            if candidate in valid_hours:
                hour = candidate
                break

    if half_after:
        minute = 30
        hour_real = hour
        textfix = re.sub(r"\b\d{1,2}\s+setengah\b",f"jam {hour_real:02d}:{minute:02d}",text)
    elif half_before:
        minute = 30
        hour_real = hour-1
        textfix = re.sub(r"\bsetengah\s+\d{1,2}\b",f"jam {hour_real:02d}:{minute:02d}",text)
    else:
        minute = 00
        hour_real = hour
        textfix = re.sub(r"\bjam\s+\d{1,2}\b",f"jam {hour_real:02d}:{minute:02d}",text)

    if date_check:
        text_date = re.sub(r"\bjam\s+\d{2}:\d{2}\b",date_check.group(0),textfix)

    if timerange:
        return {
            "texttime" : textfix.replace(timerange, "").strip(), 
            "textdate" : text_date.replace(timerange, "").strip() if date_check else None, 
            "has_time" : has_time
        }
    else:
        return {
            "texttime" : textfix.strip(), 
            "textdate" : text_date.strip() if date_check else None, 
            "has_time" : has_time
        }

def month_normalize(text): # Replacing month numbers to month names. Run after rleative checking, given relative is False
    month_match = re.search(r"\bbulan\s+(\d{1,2})\b", text)
    if not month_match:
        return text
    month = int(month_match.group(1))
    month_name = months[month]
    textfix = re.sub(r"\bbulan\s+(\d{1,2})\b", f"bulan {month_name}",text)
    return textfix

def date_normalize(textnext): # Normalizing the date phrases to obtain the exact date in YYYY-MM-DD format. Skips if week_normalize obtains this information first. Run after week_normalize in case that fails to obtain date
    weirddate = False
    date_found = False

    week_pass = re.search(r"\b\d{4}-\d{2}-\d{2}\s*\b",textnext)
    if week_pass:
        return textnext,False,True

    date_match = re.search(r"\btanggal\s+(\d{1,2})\b",textnext)
    if not date_match:
        for month in months.values():
            date_match2 = re.search(rf"\b\d{{1,2}}\s+{month}\b",textnext)
            if date_match2:
                break
        if not date_match2:
            return textnext,weirddate,date_found

    if date_match:
        day = int(date_match.group(1))
    elif date_match2:
        for month in months.values():
            monthcheck = re.search(rf"\b {month}\b",date_match2.group(0))
            if monthcheck:
                day = re.sub(rf"\b {month}\b","",date_match2.group(0))
                day = int(day)
                break
    now = datetime.fromisoformat(real_time()["datetime"])

    year_match = re.search(r"\b(?:tahun\s+)?(\d{4})\b", textnext)

    if year_match:
        year = int(year_match.group(1))
    else:
        year = now.year

    month = None
    for name,number in months_name.items():
        if name in textnext.lower():
            month = number
            break
    if month is None:
        month = now.month

    try:
        candidate = datetime(year,month,day)
    except ValueError:
        weirddate = True
        return textnext,weirddate,date_found

    if (year == now.year and candidate.date() < now.date()):
        candidate += relativedelta(years=1)

    if (month == now.month and year == now.year and candidate.date() < now.date()):
        candidate += relativedelta(months=1)

    date_text = candidate.strftime("%Y-%m-%d")

    if re.search(r"\b\d{4}\b",textnext):
        textnext = re.sub(r"\b\d{4}\b","",textnext)

    for month in months.values():
        if re.search(rf"\b{month}\b",textnext):
            textnext = re.sub(rf"\b{month}\b","",textnext)

    textfix = re.sub(rf"(?:\btanggal\s+)?{day}\b",date_text,textnext)
    date_found = True

    return textfix,weirddate,date_found

def week_normalize(text): # Normalizing the date phrase given a day name. Run after month_normalize to try getting date first
    weekday_match = re.search(r"\b(?:(?:hari|besok)\s+)?(senin|selasa|rabu|kamis|jumat|sabtu|minggu)(?:\s+(depan))?\b",text.lower())
    if not weekday_match: return text

    weekday_name = weekday_match.group(1)
    target_weekday = weekdays[weekday_name]

    now = datetime.fromisoformat(real_time()["datetime"])
    current_weekday = now.weekday()

    days_ahead = (target_weekday - current_weekday) % 7

    if weekday_match.group(2) == "depan":
        days_ahead += 7
    elif days_ahead == 0:
        days_ahead = 7

    target_date = now.date() + timedelta(days=days_ahead)
    date_text = target_date.strftime("%Y-%m-%d")

    textfix = text[:weekday_match.start()] + date_text + text[weekday_match.end():]
    return textfix

def repeat_category(text, repetitive): # Determining intervals of a repeated reminder. Run after repeat_flag
    if not repetitive:
        interval=None
        return text, interval

    text = text.lower()
    interval = None

    interval_match = re.search(r"\b(?:tiap|setiap)\s+(\d+)\s+(hari|minggu|bulan|tahun)\b",text)
    if interval_match:
        amount = int(interval_match.group(1))
        unit = interval_match.group(2)

        multiplier = {"hari" : 1, "minggu" : 7, "bulan" : 30, "tahun" : 365}

        interval = amount * multiplier[unit]
        phrase = interval_match.group(0)
    else:
        repeat_match = re.search(r"\b(?:tiap|setiap)\s+(\w+)",text)
        if repeat_match:
            word = repeat_match.group(1)
            if word == "hari":
                week_check = re.search(rf"\bhari\s+(?:{"|".join(weekdays)})\b",text)
                if week_check:
                    interval = 7
                    phrase = repeat_match.group(0)
                else:
                    interval = 1
                    phrase = repeat_match.group(0)
                # interval = 7 if re.search(rf"\bhari\s+(?:{"|".join(weekdays)})\b",text) else 1
            elif word == "minggu":
                interval = 7
                phrase = repeat_match.group(0)
            elif word == "bulan" : 
                interval = 30
                phrase = repeat_match.group(0)
            elif word == "tahun" : 
                interval = 365
                phrase = repeat_match.group(0)
            elif word in weekdays:
                interval = 7
                phrase = repeat_match.group(0)

                textfix = re.sub(rf"\b(?:tiap|setiap)\s+{word}\b",f"hari {word}",text)
                phrase = None
            elif word in ("jam", "pukul"):
                interval = 1
                phrase = re.match(r"\b(?:tiap|setiap)\b", text[repeat_match.start():]).group(0)
            elif word in fix_period:
                interval = 1
                textfix = re.sub(
                    rf"\b(?:tiap|setiap)\s+{re.escape(word)}\b",
                    f"jam {fix_period[word]}",
                    text,
                    count=1
                )
                phrase = None
            else:
                interval = 1
                phrase = repeat_match.group(0)
    if phrase:
        textfix = re.sub(phrase, "", text,count=1)
    return textfix.strip(), interval

# ==========================================
#          MEDIUM LEVEL FUNCTIONS
# ==========================================

def missing_date(text,month=None): # Run for followup functions in case a date is missing and is required
    text = text.lower()

    month_exists = any(re.search(rf"\b{month_name}\b",text) for month_name in months.values())

    text = month_normalize(text)
    text = week_normalize(text)
    text,weirddate,date_found = date_normalize(text)

    if weirddate:
        return None, True, False
    if not date_found:
        return None, False, False
    
    result = search_dates(text,languages=["id"],settings=timeset)
    if not result:
        return None, False, False

    date_dt = result[0][1]

    if month is not None and not month_exists:
        date_dt = date_dt.replace(month=month)

    return date_dt, False, True

def missing_time(text): # Run for followup functions in case the time is missing
    text = text.lower()
    timenorm = time_normalize(text)
    if not timenorm["has_time"]:
        return None, False

    result = search_dates(timenorm["texttime"],languages=["id"],settings=timeset)
    if not result:
        return None, False

    time_dt = result[0][1]
    return time_dt,True

# ==========================================
#          HIGHER LEVEL FUNCTIONS
# ==========================================

def reminder_time_normalize(text_ori): # Collect the reminder input and obtain the precise timeframe upon when the reminder is needed
    now = real_time()
    now_dt = datetime.fromisoformat(now["datetime"])
    missing = None
    timenorm = None
    text_ori = text_ori.lower()

    repetitive = repeat_flag(text_ori)
    text,interval = repeat_category(text_ori,repetitive)

    #if not repetitive:
    text,relative = relative_normalize(text)

    if relative:
        result = search_dates(text,languages=["id"],settings=timeset)[0][1].isoformat()
        date_found = True

    else:
        text = month_normalize(text)
        month_exists = any(re.search(rf"\b{month}\b",text) for month in months.values())
        text = week_normalize(text)

        text,weirddate,date_found = date_normalize(text)
        if weirddate:
            return {
                "text original": text_ori,
                "text": text,
                "repetitive": repetitive,
                "repeat_type": str(interval) + " hari",
                "result" : None,
                "relative" : relative,
                "weirddate": True,
                "date_found" : date_found
            }
        elif not date_found:
            if month_exists:
                missing = "date"
        timenorm = time_normalize(text)
        #result = search_dates(text, languages=["id"], settings=timeset)

        timeres = search_dates(timenorm["texttime"], languages=["id"],settings=timeset) if timenorm["texttime"] else None
        dateres = search_dates(timenorm["textdate"], languages=["id"],settings=timeset) if timenorm["textdate"] else None

        if dateres:
            date_dt = dateres[0][1]
        elif month_exists:
            month_dt = None
            for month, num in months_name.items():
                if re.search(rf"\b{month}\b",text):
                    month_dt = num
                    break
            if month_dt is not None:
                year = now_dt.year

                if month_dt < now_dt.month:
                    year += 1

                date_dt = now_dt.replace(year=year, month=month_dt,day=1)

            else:
                date_dt = now_dt

        else:
            date_dt = now_dt
        if timeres:
            time_dt = timeres[0][1]
        else:
            time_dt = now_dt
            if missing != "date":
                missing = "time"
                
        combined_dt = datetime.combine(date_dt.date(),time_dt.time(),tzinfo=time_dt.tzinfo)
        if combined_dt <= now_dt:
            combined_dt += relativedelta(days=1)
                
        result = combined_dt.isoformat()

        if not weirddate and timenorm["has_time"] and date_found:
            weirddate = True
            
    return {
        "text original" : text_ori,
        "text" : text,
        "repetitive" : repetitive,
        "repeat_type" : interval,
        "result" : result,
        "relative" : relative,
        "weirddate" : False,
        "has_time" : timenorm["has_time"] if timenorm else None,
        "missing" : missing,
        "date_found" : date_found
    }

def reminder_followup(text_ori,reminder): # Followup for the reminder_time_normalize when some aspects are missing.
    text = text_ori.lower()
    text_prev = reminder["text"]
    weirddate = reminder["weirddate"]
    has_time = reminder["has_time"]
    missing = reminder["missing"]
    date_found = reminder["date_found"]
    relative = reminder["relative"]
    completion = False

    if weirddate:
        result = reminder_time_normalize(text)
        if result["weirddate"]:
            reminder["weirddate"] = True
            return reminder, completion

        new_dt = datetime.fromisoformat(result["result"])
        new_date = new_dt.date()
        new_time = new_dt.time()

        for keyword in temporal_keywords:
            text_prev = re.sub(
                rf"\b{re.escape(keyword)}\b",
                "",
                text_prev
            )
        text_prev = re.sub(r"\s+", " ", text_prev).strip()

        reminder["text"] = f"{text_prev} {new_date} {new_time}"
        reminder["result"] = result["result"]
        reminder["weirddate"] = False
        reminder["has_time"] = result["has_time"]
        reminder["missing"] = result["missing"]
        reminder["date_found"] = result["date_found"]
        reminder["relative"] = result["relative"]

        return reminder, completion

    if not relative:
        if not date_found:
            missing = "date"
        elif not has_time:
            missing = "time"
        else:
            missing = None
        reminder["missing"] = missing

    if relative:
        prev_dt = datetime.fromisoformat(reminder["result"])
        now_dt = datetime.fromisoformat(real_time()["datetime"])

        if prev_dt.hour != now_dt.hour:
            missing = None
            relative = False
        elif prev_dt.day != now_dt.day:
            missing = "time"
        elif prev_dt.month != now_dt.month:
            missing = "date"
        elif prev_dt.year != now_dt.year:
            missing = "month"

        else:
            missing = "time"

        reminder["missing"] = missing
        reminder["relative"] = relative

    if missing == "month":
        old_dt = datetime.fromisoformat(reminder["result"])

        text = month_normalize(text)
        month_dt = None

        for month, num in months_name.items():
            if re.search(rf"\b{month}\b",text):
                month_dt = num
                break

        if month_dt is None:
            reminder["missing"] = "month"
            return reminder,completion
        
        new_dt = old_dt.replace(month=month_dt)
        reminder["result"] = new_dt.isoformat()

    elif missing == "date":
        old_dt = datetime.fromisoformat(reminder["result"])

        date_dt,weirddate,date_found = missing_date(text,month = old_dt.month)
        if weirddate or not date_found:
            reminder["missing"] = "date"
            return reminder, completion

        new_dt = old_dt.replace(month=date_dt.month, day=date_dt.day)
        reminder["result"] = new_dt.isoformat()
        reminder["date_found"] = date_found

    elif missing == "time":
        old_dt = datetime.fromisoformat(reminder["result"])

        time_dt,has_time=missing_time(text)
        if not has_time:
            reminder["missing"] = "time"
            return reminder, completion

        new_dt = old_dt.replace(hour=time_dt.hour, minute = time_dt.minute)
        reminder["result"] = new_dt.isoformat()
        reminder["has_time"] = has_time

    if not relative:
        if not date_found:
            missing = "date"
        elif not has_time:
            missing = "time"
        else:
            missing = None
        reminder["missing"] = missing
        
    if not missing or (relative and reminder["missing"] == "time"):
        completion = True
    return reminder, completion