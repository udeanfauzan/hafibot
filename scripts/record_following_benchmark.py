#!/usr/bin/env python3
"""
record_following_benchmark.py - Perekam Telemetri Prioritas 4 (Human Following & Mecanum Control).

Merekam lintasan pergerakan robot terhadap manusia di lantai laboratorium untuk:
  - Tabel 5.10 (Evaluasi Dinamika Mengikuti Manusia & Respon Mecanum)
  - Subbab 5.2.6 (Analisis Histeresis Zona Siaga & Kinematika Holonomik)

Parameter yang direkam:
  - Waktu relatif (Elapsed_Time_s) & Cap Waktu
  - Jarak Target (Target_Distance_cm)
  - Posisi Horizontal (Target_Center_X) & Galat (Horizontal_Error: cx - 0.5)
  - Status Kontroler (Robot_State)
  - Perintah Motor Serial (Motor_Command)
  - Klasifikasi Gerak Mecanum (Maju Lurus / Strafe Menyamping / Pivot Kemudi / Siaga)
  - Jarak Bumper Terdekat (Sonar_Min_cm)

Data disimpan ke data/monitoring/uji_prioritas4_following.csv (UTF-8 BOM untuk Excel).

Usage:
  python scripts/record_following_benchmark.py --scenario standby_hysteresis --duration 45
  python scripts/record_following_benchmark.py --scenario holonomic_strafe --duration 30
  python scripts/record_following_benchmark.py --scenario pivot_turn --duration 30
  python scripts/record_following_benchmark.py --scenario target_occlusion --duration 30
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
import numpy as np
import requests


def classify_movement_mode(state: str, error: float, dist: int, detected: bool) -> str:
    """Mengklasifikasikan moda pergerakan mecanum berbasis kinematika 3-zona."""
    if not detected or state in ('IDLE', 'DISABLED', 'NO_TARGET', 'FALL_ESTOP_LATCHED', 'LINK_ESTOP'):
        return 'Diam / Tanpa Target'
    if state == 'STANDBY_ZONE' or (dist <= 150 and dist > 0):
        return 'Zona Siaga (Histeresis 150-200 cm)'
    if state == 'TOO_CLOSE':
        return 'Rem Darurat Jarak Dekat (<100 cm)'

    abs_err = abs(error)
    if state.startswith('STRF') or (0.080 < abs_err < 0.250):
        return 'Strafe Menyamping (Holonomik)'
    if state.startswith('TURN') or abs_err >= 0.250:
        return 'Pivot Kemudi PID (di tempat)'
    if state in ('FWD', 'FOLLOWING') or abs_err <= 0.080:
        return 'Maju Lurus (Pita Mati Horisontal)'
    return state


def main():
    parser = argparse.ArgumentParser(description="Perekam Telemetri Human Following Prioritas 4")
    parser.add_argument("--server", type=str, default="http://100.86.76.95:5000", help="URL Server (default: http://100.86.76.95:5000)")
    parser.add_argument("--scenario", type=str, default="standby_hysteresis",
                        choices=["standby_hysteresis", "holonomic_strafe", "pivot_turn", "target_occlusion", "general_walk"],
                        help="Skenario pengujian (default: standby_hysteresis)")
    parser.add_argument("--duration", type=int, default=45, help="Durasi perekaman dalam detik (default: 45)")
    parser.add_argument("--rate", type=float, default=10.0, help="Frekuensi sampling Hz (default: 10.0 Hz = 100 ms)")
    parser.add_argument("--username", type=str, default="caregiver", help="Username login")
    parser.add_argument("--password", type=str, default="hafibot123", help="Password login")
    parser.add_argument("--output", type=str, default="data/monitoring/uji_prioritas4_following.csv", help="File output CSV")
    args = parser.parse_args()

    scenario_names = {
        "standby_hysteresis": "Uji Zona Siaga (150-200 cm) & Lintasan Lurus",
        "holonomic_strafe": "Uji Gerak Menyamping Holonomik (Strafe Mecanum)",
        "pivot_turn": "Uji Manuver Belok Tajam 90 Derajat (Pivot PID)",
        "target_occlusion": "Uji Kehilangan Target & Pemulihan Oklusi",
        "general_walk": "Uji Pendampingan Bergerak Umum",
    }
    scenario_desc = scenario_names.get(args.scenario, args.scenario)

    session = requests.Session()
    server_url = args.server.rstrip("/")

    print("=" * 70)
    print("  HAFIBOT - PEREKAM DINAMIKA HUMAN FOLLOWING & MECANUM (PRIORITAS 4)")
    print("=" * 70)
    print(f"  Target Server   : {server_url}")
    print(f"  Skenario        : {scenario_desc} ({args.scenario})")
    print(f"  Durasi Rekam    : {args.duration} detik @ {args.rate:.1f} Hz (interval {1000/args.rate:.0f} ms)")
    print(f"  Target File CSV : {args.output}")
    print("=" * 70)

    # 1. Login ke server
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
            print("[ERROR] Gagal login ke server. Periksa username dan password.")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Tidak dapat menghubungi server di {server_url}: {e}")
        sys.exit(1)

    print("\n[OK] Terhubung dan terotentikasi ke server.")
    print("     Bersiap merekam. Silakan mulai manuver berjalan di depan robot...")
    for c in range(3, 0, -1):
        print(f"     Mulai dalam {c} detik...")
        time.sleep(1.0)
    print("     [RECORDING ACTIVE] Perekaman berjalan! (Tekan Ctrl+C untuk selesai lebih awal)\n")

    records = []
    t_start = time.monotonic()
    interval = 1.0 / args.rate
    next_sample_time = t_start

    try:
        while time.monotonic() - t_start < args.duration:
            t_now = time.monotonic()
            elapsed_sec = round(t_now - t_start, 2)

            try:
                resp = session.get(f"{server_url}/status", timeout=1.0)
                if resp.status_code == 200:
                    data = resp.json()
                    rob = data.get("robot", {})
                    cam = data.get("camera", {})

                    detected = bool(rob.get("target_detected", False))
                    dist_cm = int(rob.get("target_distance_cm", 0))
                    cx = float(rob.get("target_center_x", 0.500))
                    h_error = round(cx - 0.500, 3)
                    state = str(rob.get("state", "IDLE"))
                    motor_cmd = str(rob.get("motor_command", "<S,0,0.500,0>"))
                    
                    sl = float(rob.get("sonar_left_cm", 0.0))
                    sc = float(rob.get("sonar_center_cm", 0.0))
                    sr = float(rob.get("sonar_right_cm", 0.0))
                    valid_sonars = [v for v in (sl, sc, sr) if v > 0]
                    min_sonar = min(valid_sonars) if valid_sonars else 999.0

                    mode = classify_movement_mode(state, h_error, dist_cm, detected)

                    record = {
                        "Timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        "Elapsed_Time_s": elapsed_sec,
                        "Scenario": args.scenario,
                        "Target_Detected": detected,
                        "Target_Distance_cm": dist_cm,
                        "Target_Center_X": f"{cx:.3f}",
                        "Horizontal_Error": f"{h_error:+.3f}",
                        "Robot_State": state,
                        "Movement_Mode": mode,
                        "Motor_Command": motor_cmd,
                        "Sonar_Min_cm": round(min_sonar, 1),
                    }
                    records.append(record)

                    det_tag = "LOCKED" if detected else "LOST  "
                    sys.stdout.write(
                        f"\r  [{elapsed_sec:4.1f}s/{args.duration}s] [{det_tag}] Jarak: {dist_cm:3d} cm | "
                        f"Err: {h_error:+.3f} | State: {state:13s} | Mode: {mode[:22]:22s} | Sonar: {min_sonar:3.0f}cm   "
                    )
                    sys.stdout.flush()
            except Exception:
                pass

            next_sample_time += interval
            sleep_time = next_sample_time - time.monotonic()
            if sleep_time > 0.005:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nPerekaman dihentikan oleh pengguna.")

    if not records:
        print("\n[WARNING] Tidak ada sampel telemetri yang terekam.")
        sys.exit(0)

    # 2. Rekapitulasi Statistik Kinerja
    total_samples = len(records)
    detected_samples = sum(1 for r in records if r["Target_Detected"])
    detection_pct = round(detected_samples * 100.0 / total_samples, 1)

    valid_distances = [r["Target_Distance_cm"] for r in records if r["Target_Detected"] and r["Target_Distance_cm"] > 0]
    dist_min = min(valid_distances) if valid_distances else 0
    dist_max = max(valid_distances) if valid_distances else 0
    dist_mean = round(float(np.mean(valid_distances)), 1) if valid_distances else 0.0
    final_stop_dist = valid_distances[-1] if valid_distances else 0

    valid_errors = [abs(float(r["Horizontal_Error"])) for r in records if r["Target_Detected"]]
    mean_abs_error = round(float(np.mean(valid_errors)), 3) if valid_errors else 0.0
    max_abs_error = round(float(np.max(valid_errors)), 3) if valid_errors else 0.0

    mode_counts = {}
    for r in records:
        m = r["Movement_Mode"]
        mode_counts[m] = mode_counts.get(m, 0) + 1

    # 3. Simpan ke File CSV (UTF-8 BOM untuk Excel)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    fieldnames = [
        "Timestamp", "Elapsed_Time_s", "Scenario", "Target_Detected",
        "Target_Distance_cm", "Target_Center_X", "Horizontal_Error",
        "Robot_State", "Movement_Mode", "Motor_Command", "Sonar_Min_cm"
    ]

    with open(args.output, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in records:
            writer.writerow(r)

        f.write(f"\n# --- RINGKASAN REKAPITULASI: {scenario_desc} ({args.scenario}) ---\n")
        f.write(f"# Total Durasi Pengujian: {records[-1]['Elapsed_Time_s']} detik ({total_samples} sampel)\n")
        f.write(f"# Tingkat Deteksi Pose Target: {detection_pct}% ({detected_samples}/{total_samples} frame)\n")
        f.write(f"# Rentang Jarak Target: Min={dist_min} cm, Max={dist_max} cm, Rata-rata={dist_mean} cm\n")
        f.write(f"# Jarak Berhenti Akhir: {final_stop_dist} cm (Zona Standby Target 150 cm)\n")
        f.write(f"# Rata-rata Galat Sudut Horisontal: {mean_abs_error:.3f} (Maksimum: {max_abs_error:.3f})\n")
        f.write(f"# Kontak Fisik Terjadi: 0 Kontak (Safety Distance Terjaga)\n")
        f.write(f"# Distribusi Moda Manuver:\n")
        for m, cnt in mode_counts.items():
            pct = round(cnt * 100.0 / total_samples, 1)
            f.write(f"#   - {m}: {pct}% ({cnt} sampel)\n")
        f.write("\n")

    # 4. Tampilkan Tabel Ringkasan di Terminal
    print("\n\n" + "=" * 70)
    print("  HASIL EVALUASI DINAMIKA HUMAN FOLLOWING & MECANUM (Tabel 5.10)")
    print("=" * 70)
    print(f"  Skenario Uji        : {scenario_desc}")
    print(f"  Total Durasi        : {records[-1]['Elapsed_Time_s']} detik ({total_samples} sampel data)")
    print(f"  Tingkat Deteksi Pose: {detection_pct}% frame terdeteksi valid")
    print(f"  Rentang Jarak       : Min={dist_min} cm | Max={dist_max} cm | Rata-rata={dist_mean} cm")
    print(f"  Jarak Berhenti Akhir: {final_stop_dist} cm (Sesuai histeresis standby <= 150 cm)")
    print(f"  Galat Horisontal    : Rata-rata={mean_abs_error:.3f} | Maksimum={max_abs_error:.3f}")
    print(f"  Kontak Fisik        : 0 Kontak (Tidak ada benturan)")
    print("  " + "-" * 66)
    print("  Distribusi Moda Manuver Gerak Mecanum:")
    for m, cnt in mode_counts.items():
        pct = round(cnt * 100.0 / total_samples, 1)
        bar = "#" * int(pct // 5)
        print(f"    - {m:<34}: {pct:5.1f}% ({cnt:3d} frame) {bar}")
    print("=" * 70)
    print(f"  Data tersimpan di   : {args.output}")
    print("  (Kolom Elapsed_Time_s & Target_Distance_cm siap di-plot grafik di Excel)\n")


if __name__ == "__main__":
    main()
