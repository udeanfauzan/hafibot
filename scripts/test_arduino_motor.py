#!/usr/bin/env python3
'''
test_arduino_motor.py - Direct Serial Motor & Sonar Diagnostics for Hafibot.

Run this script directly on Jetson Nano to test Arduino Mega & motor drivers:
  python3 scripts/test_arduino_motor.py
  python3 scripts/test_arduino_motor.py --port /dev/ttyUSB0 --drive
'''

import argparse
import os
import sys
import time

try:
    import serial
except ImportError:
    print('[ERROR] pyserial belum terinstall. Jalankan: pip3 install pyserial') 
    sys.exit(1)


def find_serial_ports():
    candidates = [
        '/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3',
        '/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3',
        '/dev/ttyTHS1', '/dev/ttyTHS2',
    ]
    found = [p for p in candidates if os.path.exists(p)]
    return found


def main():
    parser = argparse.ArgumentParser(description='Test Arduino Mega Serial, Sonar, & Motors')
    parser.add_argument('--port', type=str, default=None, help='Serial port (default: auto-detect)')
    parser.add_argument('--baud', type=int, default=115200, help='Baudrate (default: 115200)')
    parser.add_argument('--drive', action='store_true', help='Perform brief 2-second forward motor drive test')
    args = parser.parse_args()

    print('=' * 60)
    print('  HAFIBOT - Direct Arduino Motor & Sonar Diagnostic')
    print('=' * 60)

    ports = [args.port] if args.port else find_serial_ports()
    if not ports:
        print('[ERROR] Tidak ada port serial terdeteksi (/dev/ttyUSB* atau /dev/ttyACM*).')
        print('       Langkah pengecekan hardware USB di Jetson Nano:')
        print('       1. Cek apakah chip USB terdeteksi hardware:')
        print('          lsusb')
        print('       2. Cek pesan kernel Linux saat kabel USB dicolok:')
        print('          dmesg | tail -n 20')
        print('       3. Pastikan driver kernel CH340 / FTDI / ACM aktif:')
        print('          sudo modprobe ch341 ftdi_sio cdc_acm')
        print('       4. Coba lepas dan pasang kembali kabel USB Arduino ke port USB Jetson yang lain.')
        sys.exit(1)

    selected_port = ports[0]
    print(f'Detected ports : {ports}')
    print(f'Connecting to  : {selected_port} @ {args.baud} baud...')

    try:
        conn = serial.Serial(selected_port, args.baud, timeout=0.5, write_timeout=0.5)
    except Exception as e:
        print(f'[ERROR] Gagal membuka {selected_port}: {e}')
        print('       Jika permission denied, jalankan:')
        print(f'         sudo usermod -a -G dialout $USER && sudo chmod 666 {selected_port}')
        sys.exit(1)

    # Wait for Arduino bootloader reset
    print('Waiting for Arduino bootloader (2 detik)...')
    time.sleep(2.0)
    conn.reset_input_buffer()

    print('\n[*] Membaca telemetri live dari Arduino (Tekan Ctrl+C untuk stop)...')
    print('-' * 60)
    print(f'{"Timestamp":>10s} | {"State":12s} | {"Sonar Left":10s} | {"Sonar Center":12s} | {"Sonar Right":11s} | {"ESTOP"}')
    print('-' * 60)

    # Reset ESTOP command first
    conn.write(b'<R,0,0.500,0>\n')
    time.sleep(0.1)

    try:
        for _ in range(15):
            line = conn.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            if line.startswith('S,'):
                parts = line.split(',')
                if len(parts) >= 6:
                    state = parts[1]
                    us_l = parts[2]
                    us_c = parts[3]
                    us_r = parts[4]
                    estop = 'LATCHED' if parts[5] == '1' else 'OK'
                    print(f'{time.strftime("%H:%M:%S"):>10s} | {state:12s} | {us_l:>8s} cm | {us_c:>10s} cm | {us_r:>9s} cm | {estop}')
            elif line.startswith('#'):
                print(f'  [INFO] {line}')
            time.sleep(0.1)

        if args.drive:
            print('\n' + '=' * 60)
            print('  [*] MENJALANKAN TEST MOTOR FORWARD (2 DETIK)...') 
            print('  Pastikan robot diangkat atau di area aman!')
            print('=' * 60)
            t_end = time.time() + 2.0
            while time.time() < t_end:
                conn.write(b'<F,380,0.500,1>\n')
                line = conn.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('S,'):
                    parts = line.split(',')
                    print(f'  [DRIVE LIVE] State: {parts[1]:8s} | Motor Forward Active') 
                time.sleep(0.1)

            print('\n[*] MENGIRIM STOP COMMAND...')
            conn.write(b'<S,0,0.500,0>\n')
            time.sleep(0.5)
            conn.write(b'<S,0,0.500,0>\n')
            print('[*] Test selesai. Robot berhenti.')
        else:
            print('\nTips: Jalankan dengan --drive untuk menguji pergerakan roda:')
            print(f'  python3 scripts/test_arduino_motor.py --port {selected_port} --drive')

    except KeyboardInterrupt:
        print('\n\n[*] Dihentikan user.')
    finally:
        conn.write(b'<S,0,0.500,0>\n')
        conn.close()
        print('[*] Serial ditutup.')

if __name__ == '__main__':
    main()
