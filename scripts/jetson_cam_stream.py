#!/usr/bin/env python3
"""
Jetson Nano CSI Camera → HTTP MJPEG Streamer
=============================================
Jalankan script ini di Jetson Nano agar kamera CSI bisa diakses
dari laptop via jaringan WiFi untuk keperluan kalibrasi.

Penggunaan (di Jetson):
    python3 scripts/jetson_cam_stream.py

Lalu di laptop, jalankan:
    python scripts/calibrate_shoulder_distance.py --source http://<IP_JETSON>:8090/video

Resolusi output HARUS SAMA dengan yang dipakai fall_detection_node.py (640x480)
agar koefisien kalibrasi langsung kompatibel.
"""

import cv2
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# --- Konfigurasi ---
DEFAULT_PORT = 8090
# Resolusi output HARUS match dengan fall_detection_node.py!
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def gstreamer_pipeline():
    """Pipeline GStreamer yang SAMA dengan fall_detection_node.py."""
    return (
        "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
        "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, "
        "format=(string)NV12, framerate=(fraction)30/1 ! "
        "queue max-size-buffers=2 leaky=downstream ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw, width=(int){FRAME_WIDTH}, height=(int){FRAME_HEIGHT}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )


class MJPEGHandler(BaseHTTPRequestHandler):
    """Handler HTTP yang menyajikan MJPEG stream."""
    camera = None  # Akan di-set dari luar

    def do_GET(self):
        if self.path == '/video':
            self.send_response(200)
            self.send_header('Content-type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            while True:
                if MJPEGHandler.camera is None:
                    break
                ret, frame = MJPEGHandler.camera.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                ret, jpeg = cv2.imencode('.jpg', frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue

                try:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpeg)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                except BrokenPipeError:
                    break

                time.sleep(1.0 / 15)  # ~15 FPS

        elif self.path == '/':
            # Halaman info sederhana
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""<html><body style="font-family:monospace; background:#1a1a2e; color:#eee; padding:40px;">
            <h1>🤖 Hafibot CSI Camera Stream</h1>
            <p>Resolusi: {FRAME_WIDTH}x{FRAME_HEIGHT}</p>
            <p>Stream URL: <code>http://&lt;IP_JETSON&gt;:{DEFAULT_PORT}/video</code></p>
            <img src="/video" style="border:2px solid #0f3460; border-radius:8px;" />
            </body></html>"""
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress per-request log spam
        pass


def main():
    ap = argparse.ArgumentParser(description="Stream CSI camera via HTTP MJPEG")
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    print("=" * 55)
    print("  Hafibot CSI Camera HTTP Streamer")
    print("=" * 55)
    print(f"  Resolusi : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Port     : {args.port}")
    print()

    # Buka kamera CSI via GStreamer
    pipeline = gstreamer_pipeline()
    print(f"  GStreamer pipeline:\n  {pipeline}")
    print()

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ Gagal membuka kamera CSI!")
        print("   Pastikan kamera terpasang dan tidak dipakai proses lain.")
        return

    print("✅ Kamera CSI berhasil dibuka!")

    # Set kamera untuk handler
    MJPEGHandler.camera = cap

    # Dapatkan IP lokal
    import socket
    hostname = socket.gethostname()
    try:
        ip = socket.gethostbyname(hostname)
    except Exception:
        ip = "<?>"

    print()
    print(f"  🌐 Stream tersedia di:")
    print(f"     http://{ip}:{args.port}/video")
    print()
    print(f"  Di laptop, jalankan:")
    print(f"     python scripts/calibrate_shoulder_distance.py \\")
    print(f"         --source http://{ip}:{args.port}/video")
    print()
    print("  Tekan Ctrl+C untuk berhenti.")
    print("=" * 55)

    server = HTTPServer(('0.0.0.0', args.port), MJPEGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Streamer dihentikan.")
    finally:
        cap.release()
        server.server_close()


if __name__ == '__main__':
    main()
