#!/bin/bash
# ============================================================
# Hafibot - Setup Auto Display untuk Jetson Nano
# ============================================================
# Script ini dijalankan SEKALI SAJA untuk mengkonfigurasi Jetson
# agar setiap kali boot, DISPLAY dan xhost otomatis di-set.
#
# Setelah menjalankan script ini, kamu TIDAK perlu lagi
# mengetik 'export DISPLAY=:0' dan 'xhost +' setiap kali boot.
#
# Penggunaan:
#   chmod +x scripts/setup_jetson_display.sh
#   sudo ./scripts/setup_jetson_display.sh
# ============================================================

set -e

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════╗"
    echo "║  🤖 Hafibot - Setup Auto Display Jetson  ║"
    echo "╚══════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }
print_info()    { echo -e "${CYAN}ℹ️  $1${NC}"; }

# Cek apakah dijalankan sebagai root/sudo
if [ "$EUID" -ne 0 ]; then
    print_error "Script ini harus dijalankan dengan sudo!"
    echo "  Contoh: sudo ./scripts/setup_jetson_display.sh"
    exit 1
fi

# Deteksi user asli (bukan root) agar bisa edit .bashrc yang benar
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(eval echo "~${REAL_USER}")

print_header

# ============================================================
# 1. Tambahkan export DISPLAY=:0 ke ~/.bashrc
# ============================================================
echo ""
print_info "Langkah 1: Menambahkan 'export DISPLAY=:0' ke ${REAL_HOME}/.bashrc"

BASHRC="${REAL_HOME}/.bashrc"

if grep -q "export DISPLAY=:0" "$BASHRC" 2>/dev/null; then
    print_warning "'export DISPLAY=:0' sudah ada di .bashrc — dilewati."
else
    echo "" >> "$BASHRC"
    echo "# === Auto Display untuk Docker/GUI (ditambahkan oleh setup_jetson_display.sh) ===" >> "$BASHRC"
    echo "export DISPLAY=:0" >> "$BASHRC"
    print_success "'export DISPLAY=:0' berhasil ditambahkan ke .bashrc"
fi

# ============================================================
# 2. Buat systemd service agar xhost +local: jalan saat boot
# ============================================================
echo ""
print_info "Langkah 2: Membuat systemd service untuk 'xhost +local:'"

SERVICE_FILE="/etc/systemd/system/xhost-local.service"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Allow local connections to X server (xhost +local:)
After=display-manager.service
Requires=display-manager.service

[Service]
Type=oneshot
Environment=DISPLAY=:0
ExecStartPre=/bin/sleep 3
ExecStart=/usr/bin/xhost +local:
User=${REAL_USER}
RemainAfterExit=yes

[Install]
WantedBy=graphical.target
EOF

# Reload systemd dan enable service
systemctl daemon-reload
systemctl enable xhost-local.service

print_success "Systemd service 'xhost-local' berhasil dibuat dan di-enable."

# ============================================================
# 3. Jalankan xhost +local: sekarang juga (agar tidak perlu reboot)
# ============================================================
echo ""
print_info "Langkah 3: Menjalankan 'xhost +local:' sekarang..."

# Coba jalankan xhost sekarang
if su - "$REAL_USER" -c "DISPLAY=:0 xhost +local:" 2>/dev/null; then
    print_success "'xhost +local:' berhasil dijalankan."
else
    print_warning "Tidak bisa menjalankan xhost sekarang (mungkin X server belum aktif)."
    print_info "Akan otomatis berjalan setelah reboot."
fi

# ============================================================
# Selesai
# ============================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        ✅ Setup Selesai!                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
print_info "Yang sudah dikonfigurasi:"
echo "  1. export DISPLAY=:0  → otomatis di setiap terminal baru"
echo "  2. xhost +local:      → otomatis saat boot via systemd"
echo ""
print_info "Kamu TIDAK perlu lagi menjalankan:"
echo "  export DISPLAY=:0"
echo "  xhost +"
echo ""
print_warning "Disarankan reboot sekali untuk memastikan semua berjalan:"
echo "  sudo reboot"
echo ""
