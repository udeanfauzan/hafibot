#!/usr/bin/env bash
# =============================================================================
# setup_arduino_usb.sh - Bind USB-to-UART Chip (31b2:0011 / CH343 / CH340)
# Run this script on Jetson Nano to bind the Arduino Mega 2560 Pro Mini USB port.
# =============================================================================
set -euo pipefail

echo "============================================================"
echo " Setting up Arduino USB Serial Driver (VID 31b2 / 1a86)"
echo "============================================================"

# 1. Load kernel modules
sudo modprobe ch341 2>/dev/null || true
sudo modprobe cdc_acm 2>/dev/null || true
sudo modprobe ftdi_sio 2>/dev/null || true
sudo modprobe usbserial 2>/dev/null || true

# 2. Register custom USB Vendor/Product ID (31b2:0011) to usb-serial drivers
if [[ -d /sys/bus/usb-serial/drivers/ch341-uart ]]; then
  echo "31b2 0011" | sudo tee /sys/bus/usb-serial/drivers/ch341-uart/new_id 2>/dev/null || true
fi
if [[ -d /sys/bus/usb-serial/drivers/generic ]]; then
  echo "31b2 0011" | sudo tee /sys/bus/usb-serial/drivers/generic/new_id 2>/dev/null || true
fi

# 3. Create persistent UDEV rule for automatic binding on boot and hotplug
UDEV_FILE="/etc/udev/rules.d/99-arduino-ch34x.rules"
echo 'ACTION=="add", ATTRS{idVendor}=="31b2", ATTRS{idProduct}=="0011", RUN+="/sbin/modprobe ch341", RUN+="/bin/sh -c \'echo 31b2 0011 > /sys/bus/usb-serial/drivers/ch341-uart/new_id\'", MODE="0666", GROUP="dialout"' | sudo tee "$UDEV_FILE" >/dev/null
echo 'ACTION=="add", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666", GROUP="dialout"' | sudo tee -a "$UDEV_FILE" >/dev/null

# 4. Reload udev rules and trigger
sudo udevadm control --reload-rules 2>/dev/null || true
sudo udevadm trigger 2>/dev/null || true
sudo usermod -a -G dialout "$USER" 2>/dev/null || true

sleep 1

echo "============================================================"
echo " Checking detected serial ports:"
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "[!] Belum muncul. Coba cabut dan colok kembali kabel USB Arduino."
echo "============================================================"
