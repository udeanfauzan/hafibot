#!/bin/bash
# ============================================================
# Hafibot - Tailscale VPN Setup Script
# ============================================================
# Script ini membantu install dan mengkonfigurasi Tailscale
# di Jetson Nano agar guardian/pengasuh bisa monitoring
# dari mana saja secara privat dan aman.
#
# Tailscale membuat VPN mesh pribadi — hanya perangkat yang
# tergabung di jaringan Tailscale kamu yang bisa mengakses.
# Jauh lebih aman daripada Ngrok yang expose URL publik.
#
# Penggunaan:
#   chmod +x scripts/setup_tailscale.sh
#   sudo ./scripts/setup_tailscale.sh --install   # Install Tailscale
#   sudo ./scripts/setup_tailscale.sh --up         # Aktifkan & login
#   ./scripts/setup_tailscale.sh --status           # Cek status & IP
#   ./scripts/setup_tailscale.sh --help             # Bantuan
#
# Setelah aktif, guardian buka:
#   http://<tailscale-ip>:5000
#   atau http://jetson.tail-xxxxx.ts.net:5000 (MagicDNS)
# ============================================================

set -e

HAFIBOT_PORT=${HAFIBOT_PORT:-5000}

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════╗"
    echo "║   🤖 Hafibot - Tailscale Private VPN Setup   ║"
    echo "╚══════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }
print_info()    { echo -e "${CYAN}ℹ️  $1${NC}"; }

# ---- Install Tailscale ----
install_tailscale() {
    echo -e "${CYAN}📦 Menginstall Tailscale...${NC}"
    echo ""

    if command -v tailscale &> /dev/null; then
        local version=$(tailscale version 2>/dev/null | head -1)
        print_warning "Tailscale sudah terinstall: ${version}"
        read -p "Install ulang? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            return 0
        fi
    fi

    # Install via official script (mendukung ARM64 Jetson Nano)
    curl -fsSL https://tailscale.com/install.sh | sh

    echo ""
    print_success "Tailscale berhasil diinstall!"
    print_info "Jalankan: sudo $0 --up  untuk mengaktifkan"
}

# ---- Aktifkan Tailscale ----
tailscale_up() {
    if ! command -v tailscale &> /dev/null; then
        print_error "Tailscale belum terinstall! Jalankan: sudo $0 --install"
        exit 1
    fi

    echo -e "${CYAN}🔗 Mengaktifkan Tailscale...${NC}"
    echo ""

    # Pastikan service berjalan
    sudo systemctl enable --now tailscaled 2>/dev/null || true

    # Login & aktifkan
    # --accept-dns=false agar tidak mengganggu DNS lokal ROS
    sudo tailscale up --accept-dns=false

    echo ""
    print_success "Tailscale aktif!"
    echo ""

    show_status
}

# ---- Cek Status ----
show_status() {
    if ! command -v tailscale &> /dev/null; then
        print_error "Tailscale belum terinstall!"
        exit 1
    fi

    local status=$(tailscale status --json 2>/dev/null)
    if [ $? -ne 0 ]; then
        print_error "Tailscale belum aktif. Jalankan: sudo $0 --up"
        return 1
    fi

    local ts_ip=$(tailscale ip -4 2>/dev/null)
    local ts_hostname=$(tailscale status --self --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName','').rstrip('.'))" 2>/dev/null || echo "(tidak tersedia)")

    echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║           📊 Tailscale Status                ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  Tailscale IP : ${GREEN}${ts_ip}${NC}"
    echo -e "${BOLD}║${NC}  MagicDNS     : ${GREEN}${ts_hostname}${NC}"
    echo -e "${BOLD}║${NC}  Hafibot Port : ${CYAN}${HAFIBOT_PORT}${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  🌐 Guardian bisa akses monitoring via:"
    echo -e "${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  ${GREEN}http://${ts_ip}:${HAFIBOT_PORT}${NC}"
    if [ "$ts_hostname" != "(tidak tersedia)" ] && [ -n "$ts_hostname" ]; then
        echo -e "${BOLD}║${NC}  ${GREEN}http://${ts_hostname}:${HAFIBOT_PORT}${NC}"
    fi
    echo -e "${BOLD}║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║${NC}  📱 Guardian harus:"
    echo -e "${BOLD}║${NC}  1. Install Tailscale di HP/laptop"
    echo -e "${BOLD}║${NC}  2. Login dengan akun yang sama"
    echo -e "${BOLD}║${NC}     (atau diundang via Tailscale admin)"
    echo -e "${BOLD}║${NC}  3. Buka URL di atas"
    echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"

    echo ""
    print_info "Daftar perangkat di jaringan:"
    tailscale status 2>/dev/null
}

# ---- Matikan Tailscale ----
tailscale_down() {
    print_info "Mematikan Tailscale..."
    sudo tailscale down
    print_success "Tailscale dimatikan."
}

# ---- HTTPS Serve (untuk Web Push Notification) ----
tailscale_serve() {
    if ! command -v tailscale &> /dev/null; then
        print_error "Tailscale belum terinstall!"
        exit 1
    fi

    local ts_ip=$(tailscale ip -4 2>/dev/null)
    if [ -z "$ts_ip" ]; then
        print_error "Tailscale belum aktif. Jalankan: sudo $0 --up"
        exit 1
    fi

    echo -e "${CYAN}🔒 Mengaktifkan HTTPS reverse proxy...${NC}"
    echo ""

    # Aktifkan Tailscale Serve: HTTPS → HTTP localhost:5000
    if sudo tailscale serve --bg http://127.0.0.1:${HAFIBOT_PORT} 2>/dev/null; then
        true
    else
        sudo tailscale serve https / http://127.0.0.1:${HAFIBOT_PORT}
    fi

    echo ""
    print_success "HTTPS Serve aktif!"
    echo ""

    local ts_hostname=$(tailscale status --self --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName','').rstrip('.'))" 2>/dev/null || echo "")

    echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║     🔒 HTTPS Monitoring (Web Push Ready)     ║${NC}"
    echo -e "${BOLD}╠══════════════════════════════════════════════╣${NC}"
    if [ -n "$ts_hostname" ]; then
        echo -e "${BOLD}║${NC}  ${GREEN}https://${ts_hostname}${NC}"
    fi
    echo -e "${BOLD}║${NC}"
    echo -e "${BOLD}║${NC}  Guardian bisa:"
    echo -e "${BOLD}║${NC}  1. Buka URL di atas di browser HP"
    echo -e "${BOLD}║${NC}  2. Login → klik 🔔 Aktifkan Notifikasi"
    echo -e "${BOLD}║${NC}  3. 'Install' sebagai app (Add to Home Screen)"
    echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
}

# ---- Main ----
print_header

case "${1}" in
    --install|-i)
        install_tailscale
        ;;
    --up|-u)
        tailscale_up
        ;;
    --serve)
        tailscale_serve
        ;;
    --status|-s)
        show_status
        ;;
    --down|-d)
        tailscale_down
        ;;
    --help|-h|*)
        echo "Penggunaan: $0 [opsi]"
        echo ""
        echo "Opsi:"
        echo "  --install, -i   Install Tailscale (perlu sudo)"
        echo "  --up, -u        Aktifkan Tailscale & login (perlu sudo)"
        echo "  --serve         Aktifkan HTTPS reverse proxy (perlu sudo, wajib untuk push notif)"
        echo "  --status, -s    Cek status, IP, dan URL monitoring"
        echo "  --down, -d      Matikan Tailscale (perlu sudo)"
        echo "  --help, -h      Tampilkan bantuan"
        echo ""
        echo "Environment Variables:"
        echo "  HAFIBOT_PORT    Port Flask/Waitress (default: 5000)"
        echo ""
        echo -e "${BOLD}Setup pertama kali:${NC}"
        echo "  1. sudo $0 --install"
        echo "  2. sudo $0 --up       (akan muncul link login)"
        echo "  3. Buka link login di browser, masuk dengan akun"
        echo "  4. sudo $0 --serve    (aktifkan HTTPS untuk push notif)"
        echo "  5. $0 --status         (lihat IP & URL)"
        echo ""
        echo -e "${BOLD}Setup di HP Guardian:${NC}"
        echo "  1. Install Tailscale (Android/iOS/Desktop)"
        echo "  2. Login dengan akun yang sama atau diundang via admin"
        echo "  3. Buka https://<tailscale-hostname> (HTTPS!)"
        echo "  4. Login → klik 🔔 Aktifkan Notifikasi"
        echo ""
        echo -e "${BOLD}Kenapa perlu HTTPS?${NC}"
        echo "  🔒 Web Push Notification membutuhkan HTTPS"
        echo "  🔑 Tailscale Serve menyediakan sertifikat TLS otomatis"
        echo "  📱 Bisa di-'install' sebagai app di HP (PWA)"
        ;;
esac

