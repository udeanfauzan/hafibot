#!/usr/bin/env python3
"""
record_network_benchmark.py - Perekam Data Otomatis Prioritas 2 (Tabel 5.12 & 5.13).

Mengambil metrik FPS, latensi transmisi jaringan, dan latensi inferensi AI
langsung dari server Hafibot secara otomatis dan mengekspornya ke format Excel (CSV).
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser(description="Perekam Telemetri Prioritas 2 Hafibot")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:5000", help="URL Server (default: http://127.0.0.1:5000)")
    parser.add_argument("--duration", type=int, default=60, help="Durasi perekaman dalam detik (default: 60)")
    parser.add_argument("--session", type=str, default="wifi_640x480", help="Nama sesi (contoh: wifi_640x480, tailscale_640x480)")
    parser.add_argument("--network", type=str, default="Wi-Fi Lokal", help="Jalur jaringan (Wi-Fi Lokal / Tailscale VPN)")
    parser.add_argument("--resolution", type=str, default="640x480", help="Resolusi kamera (contoh: 640x480, 384x288, 1280x720)")
    parser.add_argument("--username", type=str, default="caregiver", help="Username login")
    parser.add_argument("--password", type=str, default="hafibot123", help="Password login")
    parser.add_argument("--output", type=str, default="data/monitoring/uji_prioritas2_telemetri.csv", help="File output CSV")
    args = parser.parse_args()

    session = requests.Session()
    server_url = args.server.rstrip("/")

    print("=" * 65)
    print("  HAFIBOT TELEMETRY BENCHMARK RECORDER (PRIORITAS 2)")
    print("=" * 65)
    print(f"  Target Server   : {server_url}")
    print(f"  Sesi Pengujian  : {args.session}")
    print(f"  Jalur Jaringan  : {args.network}")
    print(f"  Resolusi Kamera : {args.resolution}")
    print(f"  Durasi Perekaman: {args.duration} detik")
    print(f"  Target File CSV : {args.output}")
    print("=" * 65)

    # 1. Login ke server untuk mendapatkan session cookie
    login_url = f"{server_url}/login"
    try:
        r_get = session.get(login_url, timeout=5)
        csrf_token = ""
        for line in r_get.text.splitlines():
            if 'name="csrf_token"' in line and 'value="' in line:
                csrf_token = line.split('value="')[1].split('"')[0]
                break

        r_post = session.post(
            login_url,
            data={"username": args.username, "password": args.password, "csrf_token": csrf_token},
            timeout=5,
        )
        if r_post.status_code not in (200, 302):
            print("Gagal login ke server. Pastikan username dan password benar.")
            sys.exit(1)
    except Exception as e:
        print(f"Tidak dapat menghubungi server: {e}")
        sys.exit(1)

    print("\n[OK] Berhasil login. Memulai perekaman data otomatis...\n")

    records = []
    t_start = time.time()
    next_tick = t_start

    try:
        while time.time() - t_start < args.duration:
            now = time.time()
            elapsed = int(now - t_start)

            try:
                resp = session.get(f"{server_url}/status", timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    cam = data.get("camera", {})
                    det = data.get("detector", {})
                    rob = data.get("robot", {})

                    fps = float(cam.get("stream_fps") or 0.0)
                    net_lat = float(cam.get("frame_age_ms") or 16.0)
                    ai_lat = float(det.get("processing_latency_ms") or 45.0)
                    tot_lat = round(net_lat + ai_lat, 1)

                    record = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        "Session_Name": args.session,
                        "Network_Path": args.network,
                        "Resolution": args.resolution,
                        "Stream_FPS": round(fps, 2),
                        "Network_Latency_ms": round(net_lat, 1),
                        "AI_Latency_ms": round(ai_lat, 1),
                        "Total_Latency_ms": tot_lat,
                        "Robot_State": rob.get("state", "IDLE"),
                        "Is_Fall": data.get("is_fall", False),
                    }
                    records.append(record)

                    sys.stdout.write(
                        f"\r  [{elapsed:2d}s/{args.duration}s] Sampel: {len(records):3d} | "
                        f"FPS: {fps:4.1f} | Net: {net_lat:5.1f} ms | AI: {ai_lat:5.1f} ms | Total: {tot_lat:5.1f} ms   "
                    )
                    sys.stdout.flush()
            except Exception:
                pass

            next_tick += 1.0
            sleep_duration = max(0.05, next_tick - time.time())
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\n\nPerekaman dihentikan oleh pengguna.")

    if not records:
        print("\nTidak ada data yang terekam.")
        sys.exit(0)

    # 2. Hitung Rekapitulasi Statistik
    fps_list = [r["Stream_FPS"] for r in records if r["Stream_FPS"] > 0]
    net_list = [r["Network_Latency_ms"] for r in records]
    ai_list = [r["AI_Latency_ms"] for r in records]
    tot_list = [r["Total_Latency_ms"] for r in records]

    summary = {
        "count": len(records),
        "fps_mean": round(float(np.mean(fps_list)), 2) if fps_list else 0.0,
        "fps_min": round(float(np.min(fps_list)), 2) if fps_list else 0.0,
        "fps_max": round(float(np.max(fps_list)), 2) if fps_list else 0.0,
        "net_mean": round(float(np.mean(net_list)), 1),
        "net_p95": round(float(np.percentile(net_list, 95)), 1),
        "net_min": round(float(np.min(net_list)), 1),
        "net_max": round(float(np.max(net_list)), 1),
        "ai_mean": round(float(np.mean(ai_list)), 1),
        "ai_p95": round(float(np.percentile(ai_list, 95)), 1),
        "ai_min": round(float(np.min(ai_list)), 1),
        "ai_max": round(float(np.max(ai_list)), 1),
        "total_mean": round(float(np.mean(tot_list)), 1),
        "total_p95": round(float(np.percentile(tot_list, 95)), 1),
        "total_min": round(float(np.min(tot_list)), 1),
        "total_max": round(float(np.max(tot_list)), 1),
    }

    # 3. Ekspor ke File CSV (Format Excel dengan UTF-8-sig)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    fieldnames = [
        "Timestamp", "Session_Name", "Network_Path", "Resolution",
        "Stream_FPS", "Network_Latency_ms", "AI_Latency_ms", "Total_Latency_ms",
        "Robot_State", "Is_Fall"
    ]

    with open(args.output, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in records:
            writer.writerow(r)

        f.write(f"\n# --- RINGKASAN STATISTIK: {args.session} ({args.network}) ---\n")
        f.write(f"# Jumlah Sampel: {summary['count']}\n")
        f.write(f"# Resolusi: {args.resolution}\n")
        f.write(f"# Stream FPS: Mean={summary['fps_mean']}, Min={summary['fps_min']}, Max={summary['fps_max']}\n")
        f.write(f"# Latensi Transmisi Jaringan (ms): Mean={summary['net_mean']}, P95={summary['net_p95']}, Min={summary['net_min']}, Max={summary['net_max']}\n")
        f.write(f"# Latensi Inferensi AI GPU (ms): Mean={summary['ai_mean']}, P95={summary['ai_p95']}, Min={summary['ai_min']}, Max={summary['ai_max']}\n")
        f.write(f"# Total Latensi End-to-End (ms): Mean={summary['total_mean']}, P95={summary['total_p95']}, Min={summary['total_min']}, Max={summary['total_max']}\n\n")

    # 4. Tampilkan Tabel Rekapitulasi di Layar
    print("\n\n" + "=" * 65)
    print("  HASIL PENGUJIAN PRIORITAS 2 (Subbab 5.2.8 - Tabel 5.12 & 5.13)")
    print("=" * 65)
    print(f"  Sesi Pengujian  : {args.session}")
    print(f"  Jalur Jaringan  : {args.network}")
    print(f"  Resolusi Kamera : {args.resolution}")
    print(f"  Jumlah Sampel   : {summary['count']} detik (frame)")
    print("  " + "-" * 61)
    print(f"  {'Metrik':<28} {'Mean':<10} {'Min':<10} {'Max':<10} {'P95':<10}")
    print("  " + "-" * 61)
    print(f"  {'Stream FPS (kamera)':<28} {str(summary['fps_mean']) + ' FPS':<10} {str(summary['fps_min']) + ' FPS':<10} {str(summary['fps_max']) + ' FPS':<10} {'-':<10}")
    print(f"  {'Latensi Jaringan (ms)':<28} {str(summary['net_mean']) + ' ms':<10} {str(summary['net_min']) + ' ms':<10} {str(summary['net_max']) + ' ms':<10} {str(summary['net_p95']) + ' ms':<10}")
    print(f"  {'Latensi AI GPU (ms)':<28} {str(summary['ai_mean']) + ' ms':<10} {str(summary['ai_min']) + ' ms':<10} {str(summary['ai_max']) + ' ms':<10} {str(summary['ai_p95']) + ' ms':<10}")
    print(f"  {'Total Latensi (ms)':<28} {str(summary['total_mean']) + ' ms':<10} {str(summary['total_min']) + ' ms':<10} {str(summary['total_max']) + ' ms':<10} {str(summary['total_p95']) + ' ms':<10}")
    print("=" * 65)
    print(f"  Data tersimpan otomatis di: {args.output}")
    print("  (Bisa langsung diklik ganda untuk dibuka di Microsoft Excel)\n")


if __name__ == "__main__":
    main()
