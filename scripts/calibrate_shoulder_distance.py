#!/usr/bin/env python3
"""
Kalibrasi Body/Shoulder-to-Distance untuk Human Following (Headless / SSH)
==========================================================================

Versi headless — tidak memerlukan monitor/display (cv2.imshow).
Bisa dijalankan langsung di Jetson Nano via SSH.

Tujuan:
    Menemukan koefisien polinomial orde-2 (A, B, C) yang memetakan
    skala fitur tubuh efektif (bahu + fusi torso) dalam pixel menjadi
    jarak orang ke kamera (dalam cm).

    Fusi Multi-Joint (Bahu + Torso):
      - Menghilangkan bias proporsi tubuh (bahu lebar vs sempit).
      - Invarian terhadap putaran tubuh/menghadap samping (yaw rotation).

    Rumus: distance_cm = A * effective_px^2 + B * effective_px + C

Cara pakai:
    1. Jalankan di Jetson Nano (di dalam atau di luar Docker):
       python3 scripts/calibrate_shoulder_distance.py

    2. Letakkan meteran di lantai (dari kamera hingga 5 meter).

    3. Berdiri tegak menghadap kamera di jarak yang diminta (mulai dari 100 cm).
       - Script akan menampilkan nilai shoulder_px, torso_px, dan fused_px live di terminal.
       - Jika sudah stabil, tekan ENTER untuk menyimpan sample.
       - Ketik 's' + ENTER untuk skip jarak tertentu.
       - Ketik 'q' + ENTER untuk selesai dan hitung koefisien.

    4. Copy nilai [A, B, C] ke arg shoulder_polyfit di fall_detection.launch.
"""

import argparse
import math
import sys
import os
import time

import cv2
import numpy as np


# ================================================================
# KONFIGURASI
# ================================================================
# Jarak-jarak kalibrasi (cm) — sesuaikan jika perlu
DEFAULT_DISTANCES = [100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]

# GStreamer pipeline untuk CSI Camera (sama dengan fall_detection_node.py)
GSTREAMER_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
    "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, "
    "format=(string)NV12, framerate=(fraction)30/1 ! "
    "queue max-size-buffers=2 leaky=downstream ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! "
    "appsink max-buffers=1 drop=true sync=false"
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Kalibrasi body/shoulder-to-distance (headless, untuk SSH)"
    )
    ap.add_argument(
        '--source', default='csi',
        help='Sumber kamera: "csi" (default, GStreamer CSI), '
             '"0" atau "1" (webcam USB), atau URL stream'
    )
    ap.add_argument('--width',  type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument(
        '--distances', type=str, default=None,
        help='Daftar jarak kalibrasi (cm), pisahkan dengan koma. '
             'Contoh: "100,150,200,250,300"'
    )
    ap.add_argument(
        '--samples-per-distance', type=int, default=10,
        help='Jumlah frame yang dirata-ratakan per jarak (default: 10)'
    )
    ap.add_argument(
        '--torso-ratio', type=float, default=0.80,
        help='Rasio nominal lebar bahu terhadap tinggi torso (default: 0.80)'
    )
    return ap.parse_args()


def open_capture(source: str, w: int, h: int):
    """Buka kamera sesuai sumber."""
    if source.lower() == 'csi':
        print("[*] Membuka CSI Camera via GStreamer pipeline...")
        cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    else:
        try:
            src = int(source)
        except ValueError:
            src = source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def get_body_feature_px(frame, pose, torso_ratio: float = 0.80):
    """Deteksi pose dan kembalikan (effective_px, shoulder_px, torso_px). Return (-1, -1, -1) jika gagal."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return -1.0, -1.0, -1.0

    lms = result.pose_landmarks.landmark
    # Cek visibility bahu
    if lms[11].visibility < 0.5 or lms[12].visibility < 0.5:
        return -1.0, -1.0, -1.0

    w = frame.shape[1]
    h = frame.shape[0]

    ls_x = lms[11].x * w
    ls_y = lms[11].y * h
    rs_x = lms[12].x * w
    rs_y = lms[12].y * h

    shoulder_px = math.hypot(rs_x - ls_x, rs_y - ls_y)
    if not math.isfinite(shoulder_px) or shoulder_px <= 5.0:
        return -1.0, -1.0, -1.0

    mid_sx = (ls_x + rs_x) * 0.5
    mid_sy = (ls_y + rs_y) * 0.5

    torso_px = 0.0
    effective_px = shoulder_px

    # Fusi dengan pinggul jika terlihat
    if len(lms) > 24 and lms[23].visibility >= 0.5 and lms[24].visibility >= 0.5:
        lh_x = lms[23].x * w
        lh_y = lms[23].y * h
        rh_x = lms[24].x * w
        rh_y = lms[24].y * h

        mid_hx = (lh_x + rh_x) * 0.5
        mid_hy = (lh_y + rh_y) * 0.5

        dx_t = mid_hx - mid_sx
        dy_t = mid_hy - mid_sy
        h_torso = math.hypot(dx_t, dy_t)

        if math.isfinite(h_torso) and h_torso > 5.0 and mid_hy > mid_sy:
            torso_px = h_torso
            s_torso_equiv = torso_ratio * h_torso
            if shoulder_px < s_torso_equiv * 0.85:
                effective_px = s_torso_equiv
            else:
                effective_px = 0.5 * shoulder_px + 0.5 * s_torso_equiv

    return effective_px, shoulder_px, torso_px


def main():
    args = parse_args()

    # Tentukan daftar jarak
    if args.distances:
        distances = [int(d.strip()) for d in args.distances.split(',')]
    else:
        distances = DEFAULT_DISTANCES

    # Import MediaPipe
    try:
        import mediapipe as mp
    except ImportError:
        print("[!] mediapipe belum terinstall. pip install mediapipe")
        sys.exit(1)

    # Gunakan model_complexity=0 (Lite) sama seperti di fall_detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = open_capture(args.source, args.width, args.height)
    if not cap.isOpened():
        print(f"[!] Gagal buka kamera: {args.source}")
        sys.exit(1)

    # Warmup — biarkan kamera stabil
    print("[*] Warming up kamera (3 detik)...")
    t0 = time.time()
    while time.time() - t0 < 3.0:
        cap.read()
    print("[*] Kamera siap!\n")

    print("=" * 65)
    print("  KALIBRASI BODY/SHOULDER-TO-DISTANCE (FUSI MULTI-JOINT)")
    print("=" * 65)
    print()
    print("  Fitur Kalibrasi:")
    print("    • Fusi Bahu + Torso (Invarian terhadap lebar tubuh & putaran samping)")
    print("    • Headless (Dapat dijalankan langsung via SSH)")
    print()
    print("  Petunjuk:")
    print("    1. Letakkan meteran di lantai mulai dari kamera.")
    print("    2. Berdiri tegak menghadap kamera di jarak yang diminta.")
    print("    3. Tunggu nilai fused_px stabil (~3 detik).")
    print("    4. Tekan ENTER untuk menyimpan.")
    print("    5. Ketik 's' + ENTER untuk skip jarak ini.")
    print("    6. Ketik 'q' + ENTER untuk selesai.")
    print()
    print("=" * 65)

    samples = []  # list of (effective_px_avg, distance_cm)

    for dist_cm in distances:
        print(f"\n>>> Berdiri di jarak: {dist_cm} cm <<<")
        print("    (Tekan ENTER jika sudah stabil, 's'=skip, 'q'=selesai)\n")

        # Live display shoulder_px sampai user tekan Enter
        buffer = []  # buffer frame terakhir untuk rata-rata
        running = True

        while running:
            ok, frame = cap.read()
            if not ok:
                print("    [!] Frame gagal dibaca, mencoba ulang...")
                time.sleep(0.1)
                continue

            eff_px, spx, tpx = get_body_feature_px(frame, pose, args.torso_ratio)

            if eff_px > 0:
                buffer.append(eff_px)
                # Simpan hanya N frame terakhir
                if len(buffer) > args.samples_per_distance:
                    buffer.pop(0)
                avg = np.mean(buffer)
                std = np.std(buffer) if len(buffer) > 1 else 0
                status = "STABIL ✓" if std < 3.0 and len(buffer) >= 5 else "menunggu..."
                sys.stdout.write(
                    f"\r    shoulder={spx:5.1f}px torso={tpx:5.1f}px -> fused={eff_px:5.1f}px  "
                    f"(avg={avg:5.1f}, std={std:4.1f}, n={len(buffer):2d})  [{status}]    "
                )
                sys.stdout.flush()
            else:
                sys.stdout.write(
                    "\r    [Tidak terdeteksi — pastikan tubuh/bahu terlihat kamera]    "
                )
                sys.stdout.flush()

            try:
                import select as _sel
                if _sel.select([sys.stdin], [], [], 0.0)[0]:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input == 'q':
                        print("\n\n[*] Kalibrasi dihentikan oleh user.")
                        running = False
                        distances = []
                        break
                    elif user_input == 's':
                        print(f"\n    [→] Skip jarak {dist_cm} cm")
                        running = False
                        break
                    else:
                        if len(buffer) >= 3:
                            avg = np.mean(buffer)
                            samples.append((avg, dist_cm))
                            print(f"\n    [✓] Disimpan! fused_px = {avg:.1f} "
                                  f"@ {dist_cm} cm (dari {len(buffer)} frame)")
                        else:
                            print(f"\n    [!] Belum cukup data (min 3 frame). "
                                  "Tunggu deteksi stabil lalu tekan ENTER lagi.")
                            continue
                        running = False
                        break
            except (ImportError, OSError):
                import msvcrt  # type: ignore
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch == '\r' or ch == '\n':
                        if len(buffer) >= 3:
                            avg = np.mean(buffer)
                            samples.append((avg, dist_cm))
                            print(f"\n    [✓] Disimpan! fused_px = {avg:.1f} "
                                  f"@ {dist_cm} cm (dari {len(buffer)} frame)")
                        else:
                            print("\n    [!] Belum cukup data.")
                            continue
                        running = False
                    elif ch == 'q':
                        print("\n\n[*] Kalibrasi dihentikan.")
                        running = False
                        distances = []
                    elif ch == 's':
                        print(f"\n    [→] Skip jarak {dist_cm} cm")
                        running = False

            time.sleep(0.066)  # ~15 FPS

    cap.release()
    pose.close()

    # ================================================================
    # HITUNG POLYFIT
    # ================================================================
    if len(samples) < 3:
        print("\n[!] Butuh minimal 3 sample untuk polyfit orde-2.")
        print(f"    Saat ini hanya ada {len(samples)} sample.")
        return

    xs = np.array([s[0] for s in samples])
    ys = np.array([s[1] for s in samples])
    coeff = np.polyfit(xs, ys, 2)  # y = A*x^2 + B*x + C
    A, B, C = coeff

    print()
    print("=" * 65)
    print("  HASIL KALIBRASI")
    print("=" * 65)
    print(f"  Jumlah sample  : {len(samples)}")
    print(f"  Kamera         : {'CSI (GStreamer)' if args.source == 'csi' else args.source}")
    print("  Resolusi frame : 640 x 480")
    print(f"  Fusi Torso     : Aktif (rasio={args.torso_ratio:.2f})")
    print()
    print("  Koefisien Polyfit (orde 2):")
    print(f"    A = {A:.6f}")
    print(f"    B = {B:.6f}")
    print(f"    C = {C:.1f}")
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Copy ke fall_detection.launch:                             │")
    print(f'  │  <arg name="shoulder_polyfit"                               │')
    print(f'  │       default="[{A:.4f}, {B:.4f}, {C:.1f}]" />             │')
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    # Tabel evaluasi
    print("  Evaluasi akurasi:")
    print("  " + "-" * 50)
    print(f"  {'fused_px':>12}  {'prediksi':>10}  {'asli':>8}  {'error':>8}")
    print("  " + "-" * 50)
    total_err = 0
    for px, cm in samples:
        pred = A * px * px + B * px + C
        err = pred - cm
        total_err += abs(err)
        print(f"  {px:12.1f}  {pred:10.1f}  {cm:8d}  {err:+8.1f}")
    mae = total_err / len(samples)
    print("  " + "-" * 50)
    print(f"  Mean Absolute Error (MAE): {mae:.1f} cm")
    print()

    if mae > 30:
        print("  ⚠️  MAE tinggi (>30cm). Pertimbangkan:")
        print("      - Tambah lebih banyak sample")
        print("      - Pastikan tubuh terlihat jelas saat kalibrasi")
        print("      - Pastikan pencahayaan cukup")
    else:
        print("  ✅ Kalibrasi bagus! MAE di bawah 30cm.")


if __name__ == '__main__':
    main()

