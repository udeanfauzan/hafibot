"""
test_chatbot_integration.py - Comprehensive Unit & Integration Tests for Hafibot Voice Chatbot.
"""

import io
import json
import os
import sys
import tempfile
import time
import wave
import math
import struct
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot_service import (
    ChatbotService,
    DistilBertClassifier,
    ReminderEngine,
    DialogueGenerator,
    AudioEngine,
    get_jakarta_time,
    fetch_weather_bmkg,
    fetch_news_tempo,
)


def create_mock_wav(duration: float = 0.5, freq: float = 440.0) -> bytes:
    """Creates in-memory 16kHz mono WAV bytes."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            val = int(12000 * math.sin(2.0 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))
    return bio.getvalue()


class TestDistilBertClassifier:
    """Tests NLP classification heads for Indonesian commands and queries."""

    @pytest.fixture(scope="class")
    def classifier(self):
        return DistilBertClassifier(PROJECT_ROOT)

    def test_classifier_motion_commands(self, classifier):
        res_mulai = classifier.predict("robot mulai ikuti saya")
        assert res_mulai["intent"] in ["mulai", "ikut"]
        assert res_mulai["function"] in ["command", "chat", "response"]

        res_stop = classifier.predict("robot berhenti sekarang")
        assert res_stop["intent"] in ["berhenti", "stop"]

    def test_classifier_general_queries(self, classifier):
        res_sapa = classifier.predict("selamat pagi robot")
        assert res_sapa["intent"] in ["menyapa", "sapaan", "tanya_kabar"]

        res_jam = classifier.predict("sekarang jam berapa ya")
        assert res_jam["intent"] in ["nanya_waktu", "waktu"] or res_jam["function"] in ["time", "chat"]

    def test_classifier_empty_input(self, classifier):
        res = classifier.predict("")
        assert res["intent"] == "none"
        assert res["sentiment"] == "neutral"


class TestReminderEngine:
    """Tests persistent reminder operations and Islamic prayer time preloading."""

    @pytest.fixture
    def temp_reminder_engine(self, tmp_path):
        rem_json = tmp_path / "reminders.json"
        return ReminderEngine(rem_json)

    def test_preload_prayer_times(self, temp_reminder_engine):
        reminders = temp_reminder_engine.reminders
        assert len(reminders) >= 5
        tasks = [r["task"].lower() for r in reminders]
        assert any("subuh" in t for t in tasks)
        assert any("dzuhur" in t for t in tasks)
        assert any("ashar" in t for t in tasks)
        assert any("magrib" in t or "maghrib" in t for t in tasks)
        assert any("isya" in t for t in tasks)

    def test_add_and_query_reminder(self, temp_reminder_engine):
        from datetime import datetime, timedelta

        now = datetime.now()
        target_dt = now + timedelta(hours=2)
        item = temp_reminder_engine.add_reminder("Minum obat tensi", target_dt)
        assert item["task"] == "Minum obat tensi"
        assert item["called_upon"] is False

        resp_str, results = temp_reminder_engine.query_reminders("minum obat")
        assert len(results) >= 1
        assert "Minum obat tensi" in resp_str


class TestAudioEngine:
    """Tests TTS audio generation and audio formats."""

    @pytest.fixture
    def audio_engine(self):
        return AudioEngine()

    def test_synthesize_wav(self, audio_engine):
        text = "Halo, selamat pagi. Saya robot Hafibot."
        wav_bytes = audio_engine.synthesize_wav(text)
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 500

        # Verify valid WAV header
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2


class TestChatbotServiceIntegration:
    """Tests end-to-end ChatbotService orchestrator and robot action hooks."""

    @pytest.fixture
    def service(self):
        return ChatbotService.get_instance(PROJECT_ROOT)

    def test_motion_action_callbacks(self, service):
        actions_called = []
        service.on_start_following = lambda: actions_called.append("START_FOLLOWING")
        service.on_stop_following = lambda: actions_called.append("STOP_FOLLOWING")
        service.on_reset_estop = lambda: actions_called.append("RESET_ESTOP")

        res1 = service.process_text_interaction("robot mulai ikuti saya")
        assert res1["action"] == "START_FOLLOWING"
        assert "START_FOLLOWING" in actions_called

        res2 = service.process_text_interaction("robot berhenti")
        assert res2["action"] == "STOP_FOLLOWING"
        assert "STOP_FOLLOWING" in actions_called

        res3 = service.process_text_interaction("saya tidak apa-apa aman")
        assert res3["action"] == "RESET_ESTOP"
        assert "RESET_ESTOP" in actions_called

    def test_proactive_fall_alert(self, service):
        # Clear queue
        with service.queue_lock:
            service.proactive_audio_queue.clear()

        service.trigger_fall_alert_voice()
        item = service.pop_pending_audio()
        assert item is not None
        assert item["reason"] == "fall_confirmation"
        assert "terjatuh" in item["text"].lower() or "paimon" in item["text"].lower()
        assert len(item["wav_bytes"]) > 500


class TestServerAppVoiceEndpoints:
    """Tests Flask server voice REST API endpoints."""

    @pytest.fixture
    def client(self):
        from src.server_app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_voice_text_query_endpoint(self, client):
        resp = client.post(
            "/api/voice/text_query",
            json={"text": "Robot mulai ikuti saya"},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["action"] == "START_FOLLOWING"
        assert len(data["audio_base64"]) > 0

    def test_voice_interact_with_wav_bytes(self, client):
        mock_wav = create_mock_wav(0.2)
        resp = client.post(
            "/api/voice/interact",
            data=mock_wav,
            headers={"Content-Type": "audio/wav"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "success" in data
        assert "response_text" in data

    def test_voice_status_endpoint(self, client):
        resp = client.get("/api/voice/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ACTIVE"
        assert "reminders" in data
        assert "history" in data


    def test_voice_toggle_endpoint(self, client):
        from src.server_app import chatbot_service
        with chatbot_service.queue_lock:
            chatbot_service.proactive_audio_queue.append({
                "id": "stale-audio", "wav_bytes": b"old", "text": "old",
                "reason": "test", "priority": "normal",
            })

        # Test toggle off
        resp = client.post(
            "/api/voice/toggle",
            json={"enable": False},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["voice_enabled"] is False
        assert chatbot_service.proactive_audio_queue == []

        status = client.get("/api/voice/status").get_json()
        assert status["status"] == "DISABLED"
        assert status["voice_enabled"] is False

        disabled_query = client.post(
            "/api/voice/text_query",
            json={"text": "Halo robot"},
            headers={"Content-Type": "application/json"},
        )
        assert disabled_query.status_code == 409
        assert disabled_query.get_json()["error"] == "voice_disabled"

        pending = client.get("/api/voice/pending_audio").get_json()
        assert pending["has_audio"] is False
        assert pending["voice_enabled"] is False

        # Test toggle back on
        resp2 = client.post(
            "/api/voice/toggle",
            json={"enable": True},
            headers={"Content-Type": "application/json"},
        )
        assert resp2.status_code == 200
        data2 = resp2.get_json()
        assert data2["success"] is True
        assert data2["voice_enabled"] is True
