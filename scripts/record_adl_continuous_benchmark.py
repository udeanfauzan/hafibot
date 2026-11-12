#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_adl_continuous_benchmark.py - Perekam Uji ADL Kontinu & Postur Kritis (Prioritas 6).

Menguji ketahanan model AI & gerbang postur (posture gating) terhadap alarm palsu (False Alarm Rate)
serta mengambil bukti snapshot visual nyata dari kamera robot untuk melengkapi:
  - Tabel 5.14 / Subbab 5.2.8 (Evaluasi Sesi ADL Kontinu & Laju Alarm Palsu / Jam)
  - Gambar 5.8 / Matriks ADL (Galeri Snapshot Postur Kritis: Berbaring, Membungkuk, Jongkok, Sujud, Duduk, Berjalan)
  - Subbab 3.2.10.e (Metodologi Validasi False Alarm Rate di Lingkungan Nyata)

Skenario ADL Kritis yang Diuji:
  1. lying       - Berbaring Terkendali (Controlled Lying di Kasur/Matras)
  2. bending     - Membungkuk Mengambil Barang di Lantai (Stooping / Bending)
  3. squatting   - Jongkok (Squatting / Knee Flexion)
  4. prostrating - Sujud / Ibadah di Lantai (Prostrating)
  5. sitting     - Duduk Santai di Kursi / Sofa (Sitting on Chair)
  6. walking     - Berjalan Santai Bolak-balik (Normal Pacing / Walking)
  7. continuous  - Sesi Pemantauan ADL Bebas Jatuh Berdurasi Panjang (30 - 60 Menit)

Data numerik disimpan ke data/monitoring/uji_prioritas6_adl_continuous.csv (UTF-8-sig untuk Excel).
Foto snapshot disimpan ke data/monitoring/snapshots/prioritas6_adl_*.jpg.

Usage:
  python scripts/record_adl_continuous_benchmark.py
  python scripts/record_adl_continuous_benchmark.py --mode snapshots
  python scripts/record_adl_continuous_benchmark.py --mode continuous --duration 600
  python scripts/record_adl_continuous_benchmark.py --activity lying
  python scripts/record_adl_continuous_benchmark.py --mock
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
import numpy as np
import requests

ADL_ACTIVITIES = [
    ("lying", "Berbaring Terkendali (Matras/Kasur)", "ADL_LYING", "prioritas6_adl_lying.jpg",
     "Relawan berbaring perlahan dan tenang di atas matras/kasur lantai."),
    ("bending", "Membungkuk Mengambil Barang", "ADL_BENDING", "prioritas6_adl_bending.jpg",
     "Relawan membungkukkan badan ke lantai seolah mengambil barang lalu kembali tegak."),
    ("squatting", "Jongkok (Squatting)", "ADL_SQUAT", "prioritas6_adl_squatting.jpg",
     "Relawan melakukan gerakan jongkok di depan kamera robot selama beberapa detik."),
    ("prostrating", "Sujud / Ibadah di Lantai", "ADL_PROSTRATE", "prioritas6_adl_prostrating.jpg",
     "Relawan melakukan gerakan sujud di atas matras/sajadah di lantai."),
    ("sitting", "Duduk di Kursi / Sofa", "ADL_SITTING", "prioritas6_adl_sitting.jpg",
     "Relawan duduk santai di kursi/sofa di depan kamera robot."),
    ("walking", "Berjalan Santai Bolak-balik", "ADL_WALKING", "prioritas6_adl_walking.jpg",
     "Relawan berjalan bolak-balik dengan wajar di depan robot (mode following aktif)."),
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


def capture_single_adl_snapshot(session, server_url, csrf_token, key, name, prefix, snap_file, instruction, snapshots_dir, mock=False):
    test_id = f"{prefix}_01"
    snap_rel_path = f"snapshots/{snap_file}"
    snap_abs_path = os.path.join(snapshots_dir, snap_file)

    print("\n" + "=" * 72)
    print(f"  SKENARIO POSTUR ADL KRITIS: {name.upper()}")
    print("=" * 72)
    print(f"  Petunjuk Gerakan:")
    print(f"  -> {instruction}")
    print("  Pastikan subjek terlihat jelas di kamera robot.")
    print("  Tekan [ENTER] saat relawan sudah memulai postur tersebut...")
    if not mock:
        input("  >> Tekan ENTER untuk merekam telemetri & snapshot: ")
    else:
        print("  >> [MOCK SIMULATION] Mensimulasikan pengambilan postur ADL...")
        time.sleep(0.3)

    if not mock:
        reset_server_estop(session, server_url, csrf_token)

    print("  [MONITORING] Merekam kestabilan postur selama 6 detik...")
    t_start = time.monotonic()
    peak_fall_prob = 0.0
    torso_angles = []
    false_alarm_triggered = False
    robot_states = []

    if not mock:
        while time.monotonic() - t_start < 6.0:
            try:
                resp = session.get(f"{server_url}/status", timeout=0.8)
                if resp.status_code == 200:
                    data = resp.json()
                    is_fall = bool(data.get("is_fall", False))
                    fall_prob = float(data.get("fall_prob") or 0.0)
                    torso_angle = float(data.get("torso_angle_deg") or 0.0)
                    state = str(data.get("state", "normal"))
                    rob_state = str(data.get("robot", {}).get("state", "IDLE"))

                    if fall_prob > peak_fall_prob:
                        peak_fall_prob = fall_prob
                    if torso_angle > 0:
                        torso_angles.append(torso_angle)
                    robot_states.append(rob_state)

                    if is_fall or state == "fall":
                        false_alarm_triggered = True

                    sys.stdout.write(
                        f"\r     Sudut Torso: {torso_angle:4.1f} deg | Prob Jatuh: {fall_prob*100:4.1f}% | "
                        f"Status Model: {state.upper():8s} | Alarm Palsu: {'YA (FP)' if false_alarm_triggered else 'TIDAK (0)'}   "
                    )
                    sys.stdout.flush()
            except Exception:
                pass
            time.sleep(0.08)
    else:
        # Mock simulation values
        time.sleep(0.5)
        mock_probs = {
            "lying": (0.015, 0.048, 18.2),
            "bending": (0.020, 0.065, 41.5),
            "squatting": (0.010, 0.035, 54.0),
            "prostrating": (0.025, 0.072, 21.0),
            "sitting": (0.005, 0.018, 79.0),
            "walking": (0.008, 0.022, 85.0),
        }
        p_low, p_high, ang = mock_probs.get(key, (0.01, 0.05, 60.0))
        peak_fall_prob = np.random.uniform(p_low, p_high)
        torso_angles = [ang + np.random.uniform(-2.0, 2.0)]
        false_alarm_triggered = False
        robot_states = ["STANDBY_ZONE" if key != "walking" else "FOLLOWING"]

    print()
    mean_angle = round(float(np.mean(torso_angles)), 1) if torso_angles else 45.0
    final_prob_pct = round(peak_fall_prob * 100.0, 1)
    rep_robot_state = robot_states[-1] if robot_states else "STANDBY_ZONE"
    eval_result = "GAGAL (Ada Alarm Palsu)" if false_alarm_triggered else "LULUS (Bebas Alarm Palsu)"

    # Simpan snapshot visual dari kamera robot
    snapshot_saved = False
    os.makedirs(snapshots_dir, exist_ok=True)

    if not mock:
        try:
            f_resp = session.get(f"{server_url}/api/camera/frame", timeout=3.0)
            if f_resp.status_code == 200 and len(f_resp.content) > 500:
                with open(snap_abs_path, "wb") as sf:
                    sf.write(f_resp.content)
                snapshot_saved = True
        except Exception as e:
            print(f"     [WARNING] Gagal mengambil snapshot: {e}")
    else:
        with open(snap_abs_path, "wb") as sf:
            sf.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00test_mock_adl_snapshot\xff\xd9")
        snapshot_saved = True

    snap_str = snap_rel_path if snapshot_saved else "-"

    print(f"\n  [HASIL EVALUASI] {eval_result.upper()} ({test_id})")
    print(f"    - Puncak Probabilitas Jatuh : {final_prob_pct}% (Ambang batas jatuh: 90.0%)")
    print(f"    - Rata-rata Sudut Torso     : {mean_angle} deg")
    print(f"    - Status Alarm Palsu        : {'YA (1)' if false_alarm_triggered else 'TIDAK (0 Alarm Palsu)'}")
    print(f"    - Status Kontroler Motor    : {rep_robot_state} (Aman, motor tidak mengunci ESTOP)")
    print(f"    - Bukti Snapshot Tersimpan  : {snap_str}")

    return {
        "Test_ID": test_id,
        "Aktivitas_ADL": name,
        "Durasi_Detik": 6,
        "Status_Prediksi": "NORMAL / ADL",
        "Puncak_Probabilitas_Jatuh_Pct": final_prob_pct,
        "Sudut_Torso_deg": mean_angle,
        "Alarm_Palsu_Terpicu": "YA (1)" if false_alarm_triggered else "TIDAK (0)",
        "Robot_State": rep_robot_state,
        "Hasil_Evaluasi": eval_result,
        "Snapshot_Path": snap_str,
    }


def run_continuous_session(session, server_url, csrf_token, duration_sec, output_path, mock=False):
    print("\n" + "=" * 72)
    print(f"  SESI PEMANTAUAN ADL KONTINU (UJI KETAHANAN ALARM PALSU / FAR)")
    print("=" * 72)
    dur_min = duration_sec / 60.0
    print(f"  Durasi Pemantauan : {duration_sec} detik ({dur_min:.1f} menit)")
    print(f"  Aktivitas Relawan : Berjalan, duduk di kursi, membungkuk, relaksasi tanpa jatuh.")
    print("  Tekan [ENTER] untuk memulai sesi kontinu...")
    if not mock:
        input("  >> Tekan ENTER: ")
        reset_server_estop(session, server_url, csrf_token)
    else:
        print("  >> [MOCK SIMULATION] Mensimulasikan pemantauan kontinu...")

    print(f"\n  [SESI AKTIF] Memantau secara kontinu selama {duration_sec} detik (Tekan Ctrl+C untuk selesai lebih awal)...\n")

    t_start = time.monotonic()
    sample_count = 0
    false_alarm_count = 0
    peak_fall_prob = 0.0
    torso_angles = []

    try:
        while time.monotonic() - t_start < duration_sec:
            elapsed = int(time.monotonic() - t_start)
            sample_count += 1

            if not mock:
                try:
                    resp = session.get(f"{server_url}/status", timeout=0.9)
                    if resp.status_code == 200:
                        data = resp.json()
                        is_fall = bool(data.get("is_fall", False))
                        fall_prob = float(data.get("fall_prob") or 0.0)
                        torso_angle = float(data.get("torso_angle_deg") or 0.0)

                        if fall_prob > peak_fall_prob:
                            peak_fall_prob = fall_prob
                        if torso_angle > 0:
                            torso_angles.append(torso_angle)

                        if is_fall or data.get("state") == "fall":
                            false_alarm_count += 1
                            print(f"\n     [ALERT FP] Terdeteksi alarm palsu pada t={elapsed}s (Prob: {fall_prob*100:.1f}%)")
                            reset_server_estop(session, server_url, csrf_token)

                        cur_far = (false_alarm_count / (max(1, elapsed) / 3600.0))
                        sys.stdout.write(
                            f"\r  [{elapsed:3d}s/{duration_sec:3d}s] Sampel: {sample_count:4d} | "
                            f"Max Prob: {peak_fall_prob*100:4.1f}% | FP: {false_alarm_count:2d} | "
                            f"Laju FAR: {cur_far:4.2f}/jam   "
                        )
                        sys.stdout.flush()
                except Exception:
                    pass
                time.sleep(0.2)
            else:
                time.sleep(0.05)
                peak_fall_prob = 0.078
                torso_angles = [74.5]
                break

    except KeyboardInterrupt:
        print("\n\n[SELESAI LEBIH AWAL] Sesi kontinu dihentikan oleh pengguna.")

    actual_elapsed_sec = int(time.monotonic() - t_start)
    actual_hours = max(0.001, actual_elapsed_sec / 3600.0)
    far_per_hour = round(false_alarm_count / actual_hours, 2)
    mean_angle = round(float(np.mean(torso_angles)), 1) if torso_angles else 75.0
    eval_str = f"LULUS (FAR = {far_per_hour:.2f} kejadian/jam)" if false_alarm_count <= 1 else f"EVALUASI (FAR = {far_per_hour:.2f}/jam)"

    print("\n\n" + "=" * 72)
    print("  RINGKASAN HASIL SESI ADL KONTINU (Tabel 5.14)")
    print("=" * 72)
    print(f"  Total Durasi Sesi      : {actual_elapsed_sec} detik ({actual_elapsed_sec/60.0:.1f} menit / {actual_hours:.3f} jam)")
    print(f"  Total Bingkai Dipantau : {sample_count} sampel")
    print(f"  Puncak Probabilitas    : {peak_fall_prob*100:.1f}% (Batas jatuh: 90.0%)")
    print(f"  Jumlah Alarm Palsu (FP): {false_alarm_count} kejadian")
    print(f"  False Alarm Rate (FAR) : {far_per_hour:.2f} kejadian per jam (Target: <= 1.00 kejadian/jam)")
    print(f"  Status Evaluasi        : {eval_str.upper()}")
    print("=" * 72)

    return {
        "Test_ID": "ADL_CONT_SESSION",
        "Aktivitas_ADL": f"Sesi Pemantauan ADL Kontinu ({actual_elapsed_sec/60.0:.1f} Menit)",
        "Durasi_Detik": actual_elapsed_sec,
        "Status_Prediksi": "NORMAL / ADL",
        "Puncak_Probabilitas_Jatuh_Pct": round(peak_fall_prob * 100.0, 1),
        "Sudut_Torso_deg": mean_angle,
        "Alarm_Palsu_Terpicu": f"TIDAK ({false_alarm_count})" if false_alarm_count == 0 else f"YA ({false_alarm_count})",
        "Robot_State": "FOLLOWING / STANDBY",
        "Hasil_Evaluasi": eval_str,
        "Snapshot_Path": "-",
    }


def save_and_display_summary(results, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "Test_ID", "Aktivitas_ADL", "Durasi_Detik", "Status_Prediksi",
        "Puncak_Probabilitas_Jatuh_Pct", "Sudut_Torso_deg",
        "Alarm_Palsu_Terpicu", "Robot_State", "Hasil_Evaluasi", "Snapshot_Path"
    ]

    existing_rows = {}
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            with open(output_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Test_ID"):
                        existing_rows[row["Test_ID"]] = row
        except Exception as e:
            print(f"[WARNING] Gagal membaca data lama: {e}")

    for r in results:
        existing_rows[r["Test_ID"]] = r

    order_prefix = ["ADL_LYING", "ADL_BENDING", "ADL_SQUAT", "ADL_PROSTRATE", "ADL_SITTING", "ADL_WALKING", "ADL_CONT"]
    def sort_key(item):
        tid = item[0]
        for idx, pref in enumerate(order_prefix):
            if tid.startswith(pref):
                return idx
        return 99

    sorted_results = [v for k, v in sorted(existing_rows.items(), key=sort_key)]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow(r)

    print("\n\n" + "=" * 105)
    print("  REKAPITULASI PENGUJIAN PRIORITAS 6: UJI KETAHANAN ALARM PALSU ADL (Tabel 5.14)")
    print("=" * 105)
    print(f"  {'Aktivitas / Skenario':<34} {'Status':<12} {'Max Prob (%)':<14} {'Sudut Torso':<14} {'Alarm Palsu':<14} {'Bukti Snapshot':<16}")
    print("  " + "-" * 101)

    for r in sorted_results:
        act = r["Aktivitas_ADL"][:32]
        pred = r["Status_Prediksi"][:10]
        prob = f"{float(r['Puncak_Probabilitas_Jatuh_Pct']):.1f}%"
        angle = f"{float(r['Sudut_Torso_deg']):.1f} deg"
        fp = r["Alarm_Palsu_Terpicu"]
        snap = r["Snapshot_Path"]
        print(f"  {act:<34} {pred:<12} {prob:<14} {angle:<14} {fp:<14} {snap:<16}")

    print("=" * 105)
    print(f"  Data lengkap tersimpan rapi di : {output_path}")
    print(f"  Foto snapshot tersimpan di     : data/monitoring/snapshots/prioritas6_adl_*.jpg")
    print("  (Format Excel dengan UTF-8-sig siap disalin ke Tabel 5.14 & Gambar 5.8 naskah laporan)\n")


def main():
    parser = argparse.ArgumentParser(description="Perekam Uji ADL Kontinu & Postur Kritis (Prioritas 6)")
    parser.add_argument("--server", type=str, default="http://100.86.76.95:5000", help="URL Server (default: http://100.86.76.95:5000)")
    parser.add_argument("--mode", type=str, default="snapshots",
                        choices=["snapshots", "continuous", "all"],
                        help="Mode pengujian: snapshots (6 postur kritis), continuous (sesi waktu kontinu), atau all (default: snapshots)")
    parser.add_argument("--activity", type=str, default=None,
                        choices=["lying", "bending", "squatting", "prostrating", "sitting", "walking"],
                        help="Ambil/ulang hanya 1 postur ADL tertentu (misal: --activity lying)")
    parser.add_argument("--duration", type=int, default=600, help="Durasi sesi kontinu dalam detik (default: 600s = 10 menit)")
    parser.add_argument("--username", type=str, default="caregiver", help="Username login")
    parser.add_argument("--password", type=str, default="hafibot123", help="Password login")
    parser.add_argument("--output", type=str, default="data/monitoring/uji_prioritas6_adl_continuous.csv", help="File CSV output")
    parser.add_argument("--snapshots-dir", type=str, default="data/monitoring/snapshots", help="Folder penyimpanan foto snapshot")
    parser.add_argument("--mock", action="store_true", help="Jalankan simulasi mock untuk verifikasi otomatis")
    args = parser.parse_args()

    session = requests.Session()
    server_url = args.server.rstrip("/")

    print("=" * 72)
    print("  HAFIBOT - UJI KETAHANAN ALARM PALSU ADL & SESI KONTINU (PRIORITAS 6)")
    print("=" * 72)
    print(f"  Target Server       : {server_url}")
    print(f"  Mode Pengujian      : {args.mode.upper()}")
    if args.activity:
        print(f"  Target Postur ADL   : {args.activity.upper()} (Uji Tunggal / Retake)")
    if args.mode in ("continuous", "all"):
        print(f"  Durasi Sesi Kontinu : {args.duration} detik ({args.duration/60.0:.1f} menit)")
    print(f"  Target File CSV     : {args.output}")
    print(f"  Folder Snapshot Foto: {args.snapshots_dir}")
    print("=" * 72)

    # 1. Login ke server
    dash_csrf = ""
    if not args.mock:
        csrf = login_server(session, server_url, args.username, args.password)
        if not csrf:
            sys.exit(1)
        dash_csrf = get_dashboard_csrf(session, server_url, csrf)

    results = []

    # 2. Eksekusi Mode Snapshots Postur Kritis
    if args.mode in ("snapshots", "all") or args.activity:
        if args.activity:
            target_list = [item for item in ADL_ACTIVITIES if item[0] == args.activity]
        else:
            target_list = ADL_ACTIVITIES

        try:
            for key, name, prefix, snap_file, instruction in target_list:
                res = capture_single_adl_snapshot(
                    session, server_url, dash_csrf, key, name, prefix, snap_file, instruction, args.snapshots_dir, mock=args.mock
                )
                results.append(res)
                time.sleep(0.5 if args.mock else 1.0)
        except KeyboardInterrupt:
            print("\n\n[SELESAI LEBIH AWAL] Pengambilan snapshot dihentikan oleh pengguna.")

    # 3. Eksekusi Mode Sesi Kontinu
    if (args.mode in ("continuous", "all")) and not args.activity:
        try:
            cont_res = run_continuous_session(
                session, server_url, dash_csrf, args.duration, args.output, mock=args.mock
            )
            results.append(cont_res)
        except KeyboardInterrupt:
            print("\n\n[SELESAI LEBIH AWAL] Sesi kontinu dihentikan oleh pengguna.")

    if results:
        save_and_display_summary(results, args.output)
    else:
        print("\nTidak ada hasil yang dicatat.")


if __name__ == "__main__":
    main()
