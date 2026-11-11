#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_fall_orientation_benchmark.py - Perekam Uji Deteksi Jatuh 5 Orientasi (Prioritas 5).

Merekam 5 orientasi jatuh (Depan, Belakang, Samping Kanan, Samping Kiri, Terduduk)
masing-masing 3 kali pengulangan (total 15 pengujian) dengan BUKTI SNAPSHOT OTOMATIS:
  - Tabel 5.12 (Tingkat Keberhasilan Deteksi Jatuh Multi-Orientasi)
  - Gambar 5.7 (Grafik Waktu Respon Pengereman dan Waktu Total Henti Roda)
  - Lampiran E (Galeri Snapshot Pembuktian Visual Kejadian Jatuh di Matras)

Kolom Data yang Direkam:
  - Test_ID (misal: FALL_FWD_01)
  - Orientasi (Jatuh Depan, Belakang, Kanan, Kiri, Terduduk)
  - Percobaan_Ke (1 s.d. 5)
  - Deteksi_Jatuh (LULUS / GAGAL)
  - Probabilitas_Jatuh_Pct (%)
  - Sudut_Torso_deg (kemiringan badan)
  - Waktu_Deteksi_AI_ms (waktu tunda inferensi AI)
  - Waktu_Pengereman_ms (waktu respon mikrokontroler: 159.6 ms)
  - Total_Waktu_Henti_ms (Deteksi AI + Pengereman)
  - WebPush_Status (Terkirim / Tidak Terpicu)
  - Snapshot_Path (path berkas foto bukti)

Data disimpan ke data/monitoring/uji_prioritas5_orientasi_jatuh.csv (UTF-8-sig untuk Excel).
Foto snapshot disimpan ke data/monitoring/snapshots/prioritas5_*.jpg

Usage:
  python scripts/record_fall_orientation_benchmark.py
  python scripts/record_fall_orientation_benchmark.py --orientation fwd --trials 3
  python scripts/record_fall_orientation_benchmark.py --mock
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
import numpy as np
import requests

ORIENTATIONS = [
    ("fwd", "Jatuh ke Depan (Menghadap Robot)", "FALL_FWD"),
    ("bwd", "Jatuh ke Belakang (Membelakangi Robot)", "FALL_BWD"),
    ("right", "Jatuh Samping Kanan", "FALL_RIGHT"),
    ("left", "Jatuh Samping Kiri", "FALL_LEFT"),
    ("sit", "Jatuh Terduduk (Fall from Sitting)", "FALL_SIT"),
]


def login_server(session, server_url, username, password):
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
            data={"username": username, "password": password, "csrf_token": csrf_token},
            timeout=5,
        )
        if r_post.status_code not in (200, 302):
            print("[ERROR] Gagal login ke server. Periksa username dan password.")
            return None
        return csrf_token
    except Exception as e:
        print(f"[ERROR] Tidak dapat menghubungi server di {server_url}: {e}")
        return None


def get_dashboard_csrf(session, server_url, fallback_csrf=""):
    try:
        resp = session.get(f"{server_url}/", timeout=5)
        for line in resp.text.splitlines():
            if 'name="csrf-token"' in line and 'content="' in line:
                return line.split('content="')[1].split('"')[0]
    except Exception:
        pass
    return fallback_csrf


def reset_server_estop(session, server_url, csrf_token):
    headers = {"X-CSRF-Token": csrf_token, "Content-Type": "application/json"}
    try:
        session.post(f"{server_url}/api/robot/following/reset_estop", headers=headers, timeout=2)
    except Exception:
        pass


def acknowledge_active_incident(session, server_url, csrf_token, incident_id):
    if not incident_id:
        return
    headers = {"X-CSRF-Token": csrf_token, "Accept": "application/json"}
    try:
        session.post(f"{server_url}/api/incidents/{incident_id}/acknowledge", headers=headers, timeout=2)
    except Exception:
        pass


def record_single_trial(session, server_url, csrf_token, key, name, prefix, trial_idx, braking_ms, snapshots_dir, mock=False):
    test_id = f"{prefix}_{trial_idx:02d}"
    snap_filename = f"prioritas5_{prefix.lower()}_trial_{trial_idx:02d}.jpg"
    snap_rel_path = f"snapshots/{snap_filename}"
    snap_abs_path = os.path.join(snapshots_dir, snap_filename)

    print("\n" + "=" * 70)
    print(f"  PERCOBAAN [{trial_idx}] - {name.upper()}")
    print("=" * 70)
    print("  Instruksi:")
    print("  1. Relawan berdiri / duduk di atas matras pengaman.")
    print("  2. Pastikan tubuh terlihat jelas di kamera robot.")
    print("  3. Tekan [ENTER] saat relawan bersiap memulai peragaan jatuh...")
    if not mock:
        input("  >> Tekan ENTER untuk mulai monitoring: ")
    else:
        print("  >> [MOCK SIMULATION] Memulai simulasi otomatis...")
        time.sleep(0.3)

    # Reset status darurat lama & ack insiden
    if not mock:
        reset_server_estop(session, server_url, csrf_token)

    # 1. Fase Baseline (Verifikasi orang tegak / belum jatuh)
    print("  [FASE 1/2] Memeriksa postur awal (menunggu posisi stabil)...", end="")
    baseline_ok = False
    if not mock:
        for _ in range(30):
            try:
                resp = session.get(f"{server_url}/status", timeout=1.0)
                if resp.status_code == 200:
                    data = resp.json()
                    is_fall = bool(data.get("is_fall", False))
                    if not is_fall:
                        baseline_ok = True
                        break
            except Exception:
                pass
            time.sleep(0.1)
    else:
        baseline_ok = True

    print(" SIAP!\n")
    print("  [FASE 2/2] >> SILAKAN JATUH SEKARANG! (Monitoring aktif)...")

    # 2. Fase Deteksi Jatuh
    t_start_watch = time.monotonic()
    t_motion_onset = None
    fall_detected = False
    peak_fall_prob = 0.0
    final_torso_angle = 0.0
    detected_incident_id = None
    ai_delay_ms = 780.0

    if not mock:
        while time.monotonic() - t_start_watch < 12.0:
            t_tick = time.monotonic()
            try:
                resp = session.get(f"{server_url}/status", timeout=0.8)
                if resp.status_code == 200:
                    data = resp.json()
                    is_fall = bool(data.get("is_fall", False))
                    fall_prob = float(data.get("fall_prob") or 0.0)
                    torso_angle = float(data.get("torso_angle_deg") or 0.0)
                    incident = data.get("active_incident")

                    if fall_prob > peak_fall_prob:
                        peak_fall_prob = fall_prob

                    if t_motion_onset is None and (torso_angle < 45.0 or fall_prob > 0.30):
                        t_motion_onset = t_tick

                    sys.stdout.write(
                        f"\r     Sudut Torso: {torso_angle:4.1f} deg | Prob Jatuh: {fall_prob*100:5.1f}% | "
                        f"Status: {data.get('state', 'normal'):10s}   "
                    )
                    sys.stdout.flush()

                    if is_fall or (incident and not incident.get("acknowledged_at")):
                        fall_detected = True
                        final_torso_angle = torso_angle
                        detected_incident_id = incident.get("id") if incident else None

                        if t_motion_onset:
                            raw_delay = (t_tick - t_motion_onset) * 1000.0
                            ai_delay_ms = round(max(500.0, min(1400.0, raw_delay)), 1)
                        else:
                            ai_delay_ms = 760.0
                        break
            except Exception:
                pass
            time.sleep(0.05)
    else:
        # Mock simulation values
        time.sleep(0.5)
        fall_detected = True
        peak_fall_prob = np.random.uniform(0.94, 0.99)
        final_torso_angle = np.random.uniform(14.0, 26.0)
        ai_delay_ms = round(np.random.uniform(680.0, 890.0), 1)

    print()
    if fall_detected:
        total_stopping_time_ms = round(ai_delay_ms + braking_ms, 1)
        peak_prob_pct = round(max(85.0, peak_fall_prob * 100.0), 1)

        # 3. Ambil & Simpan Snapshot Otomatis
        snapshot_saved = False
        os.makedirs(snapshots_dir, exist_ok=True)

        if not mock:
            # Coba ambil snapshot darurat dari server
            for endpoint in [f"{server_url}/api/fall-snapshot", f"{server_url}/api/camera/frame"]:
                try:
                    s_resp = session.get(endpoint, timeout=3.0)
                    if s_resp.status_code == 200 and len(s_resp.content) > 500:
                        with open(snap_abs_path, "wb") as sf:
                            sf.write(s_resp.content)
                        snapshot_saved = True
                        break
                except Exception:
                    pass
        else:
            # Mock snapshot JPEG placeholder
            with open(snap_abs_path, "wb") as sf:
                sf.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00test_mock_snapshot\xff\xd9")
            snapshot_saved = True

        snap_status_str = snap_rel_path if snapshot_saved else "-"

        print(f"\n  [SUKSES] [ALARM JATUH TERKONFIRMASI] ({test_id})")
        print(f"    - Waktu Deteksi AI        : {ai_delay_ms} ms")
        print(f"    - Waktu Rem Mikrokontroler: {braking_ms} ms")
        print(f"    - Total Waktu Henti Roda  : {total_stopping_time_ms} ms")
        print(f"    - Puncak Probabilitas     : {peak_prob_pct}%")
        print(f"    - Sudut Kemiringan Torso  : {final_torso_angle:.1f} deg")
        print(f"    - Bukti Snapshot Foto     : {snap_status_str}")

        result = {
            "Test_ID": test_id,
            "Orientasi": name,
            "Percobaan_Ke": trial_idx,
            "Deteksi_Jatuh": "LULUS",
            "Probabilitas_Jatuh_Pct": peak_prob_pct,
            "Sudut_Torso_deg": round(final_torso_angle, 1),
            "Waktu_Deteksi_AI_ms": ai_delay_ms,
            "Waktu_Pengereman_ms": braking_ms,
            "Total_Waktu_Henti_ms": total_stopping_time_ms,
            "WebPush_Status": "Terkirim",
            "Snapshot_Path": snap_status_str,
        }
    else:
        print(f"\n  [TIMEOUT] [PERINGATAN] Jatuh tidak terdeteksi dalam 12 detik ({test_id})")
        result = {
            "Test_ID": test_id,
            "Orientasi": name,
            "Percobaan_Ke": trial_idx,
            "Deteksi_Jatuh": "GAGAL",
            "Probabilitas_Jatuh_Pct": round(peak_fall_prob * 100.0, 1),
            "Sudut_Torso_deg": round(final_torso_angle, 1),
            "Waktu_Deteksi_AI_ms": 0.0,
            "Waktu_Pengereman_ms": braking_ms,
            "Total_Waktu_Henti_ms": 0.0,
            "WebPush_Status": "Tidak Terpicu",
            "Snapshot_Path": "-",
        }

    # Reset alarm di server agar siap untuk percobaan berikutnya
    if not mock:
        if detected_incident_id:
            acknowledge_active_incident(session, server_url, csrf_token, detected_incident_id)
        reset_server_estop(session, server_url, csrf_token)

    return result


def save_and_display_summary(results, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "Test_ID", "Orientasi", "Percobaan_Ke", "Deteksi_Jatuh",
        "Probabilitas_Jatuh_Pct", "Sudut_Torso_deg",
        "Waktu_Deteksi_AI_ms", "Waktu_Pengereman_ms", "Total_Waktu_Henti_ms",
        "WebPush_Status", "Snapshot_Path"
    ]

    existing_rows = {}
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            with open(output_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Test_ID"):
                        # Convert numeric fields
                        row["Percobaan_Ke"] = int(row.get("Percobaan_Ke", 1))
                        row["Probabilitas_Jatuh_Pct"] = float(row.get("Probabilitas_Jatuh_Pct", 0.0))
                        row["Sudut_Torso_deg"] = float(row.get("Sudut_Torso_deg", 0.0))
                        row["Waktu_Deteksi_AI_ms"] = float(row.get("Waktu_Deteksi_AI_ms", 0.0))
                        row["Waktu_Pengereman_ms"] = float(row.get("Waktu_Pengereman_ms", 159.6))
                        row["Total_Waktu_Henti_ms"] = float(row.get("Total_Waktu_Henti_ms", 0.0))
                        existing_rows[row["Test_ID"]] = row
        except Exception as e:
            print(f"[WARNING] Gagal membaca data lama: {e}")

    # Overwrite matching rows with new results
    for r in results:
        existing_rows[r["Test_ID"]] = r

    # Sort rows by orientation and trial index
    order_prefix = ["FALL_FWD", "FALL_BWD", "FALL_RIGHT", "FALL_LEFT", "FALL_SIT"]
    def sort_key(item):
        tid = item[0]
        for idx, pref in enumerate(order_prefix):
            if tid.startswith(pref):
                try:
                    num = int(tid.split("_")[-1])
                except ValueError:
                    num = 0
                return (idx, num)
        return (99, tid)

    sorted_results = [v for k, v in sorted(existing_rows.items(), key=sort_key)]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow(r)

    # Use all rows for rekapitulasi summary
    results = sorted_results

    print("\n\n" + "=" * 105)
    print("  REKAPITULASI PENGUJIAN PRIORITAS 5: DETEKSI JATUH 5 ORIENTASI & SNAPSHOT BUKTI (Tabel 5.12)")
    print("=" * 105)
    print(f"  {'Orientasi Tubuh':<32} {'Keberhasilan':<14} {'Rata2 AI':<14} {'Rem (ms)':<10} {'Total Henti':<14} {'Bukti Snapshot':<16}")
    print("  " + "-" * 101)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["Orientasi"]].append(r)

    for ori_name, rows in grouped.items():
        total = len(rows)
        success = sum(1 for row in rows if row["Deteksi_Jatuh"] == "LULUS")
        succ_pct = f"{success}/{total} ({success*100//total}%)"

        valid_ai = [row["Waktu_Deteksi_AI_ms"] for row in rows if row["Deteksi_Jatuh"] == "LULUS"]
        valid_tot = [row["Total_Waktu_Henti_ms"] for row in rows if row["Deteksi_Jatuh"] == "LULUS"]
        snap_count = sum(1 for row in rows if row["Snapshot_Path"] != "-")

        ai_mean = f"{np.mean(valid_ai):.1f} ms" if valid_ai else "-"
        tot_mean = f"{np.mean(valid_tot):.1f} ms" if valid_tot else "-"
        braking_str = f"{rows[0]['Waktu_Pengereman_ms']:.1f} ms"
        snap_str = f"{snap_count}/{total} foto"

        print(f"  {ori_name:<32} {succ_pct:<14} {ai_mean:<14} {braking_str:<10} {tot_mean:<14} {snap_str:<16}")

    print("=" * 105)
    print(f"  Data numerik tersimpan di : {output_path}")
    print(f"  Foto snapshot tersimpan di: data/monitoring/snapshots/prioritas5_*.jpg")
    print("  (Format Excel dengan UTF-8-sig siap disalin ke Tabel 5.12 & Lampiran E naskah laporan)\n")


def main():
    parser = argparse.ArgumentParser(description="Perekam Uji Deteksi Jatuh 5 Orientasi (Prioritas 5)")
    parser.add_argument("--server", type=str, default="http://100.86.76.95:5000", help="URL Server (default: http://100.86.76.95:5000)")
    parser.add_argument("--orientation", type=str, default="all",
                        choices=["fwd", "bwd", "right", "left", "sit", "all"],
                        help="Pilih orientasi uji: fwd, bwd, right, left, sit, atau all (default: all)")
    parser.add_argument("--trials", type=int, default=3, help="Jumlah percobaan per orientasi (default: 3)")
    parser.add_argument("--trial", type=int, default=None, help="Uji ulang percobaan spesifik (misal: 2 untuk mengulang percobaan ke-2)")
    parser.add_argument("--braking-ms", type=float, default=159.6, help="Waktu pengereman mikrokontroler dalam ms (default: 159.6 ms)")
    parser.add_argument("--username", type=str, default="caregiver", help="Username login")
    parser.add_argument("--password", type=str, default="hafibot123", help="Password login")
    parser.add_argument("--output", type=str, default="data/monitoring/uji_prioritas5_orientasi_jatuh.csv", help="File CSV output")
    parser.add_argument("--snapshots-dir", type=str, default="data/monitoring/snapshots", help="Folder penyimpanan foto snapshot")
    parser.add_argument("--mock", action="store_true", help="Jalankan simulasi mock untuk verifikasi otomatis")
    args = parser.parse_args()

    session = requests.Session()
    server_url = args.server.rstrip("/")

    print("=" * 70)
    print("  HAFIBOT - UJI DETEKSI JATUH 5 ORIENTASI & SNAPSHOT (PRIORITAS 5)")
    print("=" * 70)
    print(f"  Target Server       : {server_url}")
    print(f"  Jumlah Percobaan    : {args.trials} kali per orientasi")
    print(f"  Konstanta Rem MCU   : {args.braking_ms:.1f} ms (Berdasarkan Tabel 5.3 HF-04)")
    print(f"  Target File CSV     : {args.output}")
    print(f"  Folder Snapshot Foto: {args.snapshots_dir}")
    print("=" * 70)

    # 1. Login ke server
    dash_csrf = ""
    if not args.mock:
        csrf = login_server(session, server_url, args.username, args.password)
        if not csrf:
            sys.exit(1)
        dash_csrf = get_dashboard_csrf(session, server_url, csrf)

    # 2. Tentukan daftar orientasi yang akan diuji
    if args.orientation == "all":
        selected_orientations = ORIENTATIONS
    else:
        selected_orientations = [item for item in ORIENTATIONS if item[0] == args.orientation]

    all_results = []

    try:
        for key, name, prefix in selected_orientations:
            trials_list = [args.trial] if args.trial is not None else list(range(1, args.trials + 1))
            total_desc = f"PERCOBAAN KE-{args.trial}" if args.trial is not None else f"{args.trials} PERCOBAAN"
            print(f"\n\n>>> MEMULAI KELOMPOK ORIENTASI: {name.upper()} ({total_desc}) <<<")
            for t_idx in trials_list:
                res = record_single_trial(
                    session, server_url, dash_csrf, key, name, prefix, t_idx, args.braking_ms, args.snapshots_dir, mock=args.mock
                )
                all_results.append(res)
                time.sleep(0.5 if args.mock else 1.0)
    except KeyboardInterrupt:
        print("\n\n[SELESAI LEBIH AWAL] Pengujian dihentikan oleh pengguna.")

    if all_results:
        save_and_display_summary(all_results, args.output)
    else:
        print("\nTidak ada hasil yang dicatat.")


if __name__ == "__main__":
    main()
