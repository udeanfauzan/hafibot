#!/bin/bash
# ============================================================
# Hafibot - Start Docker (dengan auto restart nvargus-daemon)
# ============================================================
# Script wrapper untuk menjalankan Docker Compose.
# Otomatis restart nvargus-daemon agar CSI camera siap dipakai.
#
# Penggunaan:
#   chmod +x scripts/start_docker.sh
#   ./scripts/start_docker.sh              # Start container
#   ./scripts/start_docker.sh --build      # Rebuild & start
#   ./scripts/start_docker.sh --stop       # Stop container
#   ./scripts/start_docker.sh --logs       # Lihat logs
#   ./scripts/start_docker.sh --restart    # Restart container
# ============================================================

set -e

# Path ke docker-compose.yml (relatif terhadap root project)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════╗"
    echo "║       🤖 Hafibot - Docker Launcher       ║"
    echo "╚══════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }
print_info()    { echo -e "${CYAN}ℹ️  $1${NC}"; }

# ============================================================
# Restart nvargus-daemon (agar CSI camera tidak freeze/error)
# ============================================================
restart_nvargus() {
    print_info "Merestart nvargus-daemon (CSI camera)..."

    # Cek apakah nvargus-daemon ada di sistem
    if ! command -v nvargus-daemon &> /dev/null && \
       [ ! -f /usr/sbin/nvargus-daemon ] && \
       ! systemctl list-units --type=service | grep -q nvargus; then
        print_warning "nvargus-daemon tidak ditemukan — dilewati (bukan Jetson atau tidak pakai CSI camera)."
        return 0
    fi

    # Stop dulu kalau sedang jalan
    if systemctl is-active --quiet nvargus-daemon 2>/dev/null; then
        print_info "Menghentikan nvargus-daemon..."
        sudo systemctl stop nvargus-daemon
        sleep 1
    fi

    # Start ulang
    print_info "Memulai nvargus-daemon..."
    sudo systemctl restart nvargus-daemon
    sleep 2

    # Verifikasi
    if systemctl is-active --quiet nvargus-daemon 2>/dev/null; then
        print_success "nvargus-daemon berhasil di-restart dan aktif!"
    else
        print_warning "nvargus-daemon mungkin gagal start. Cek dengan: sudo systemctl status nvargus-daemon"
    fi
}

# ============================================================
# Cek apakah /dev/video0 ada (camera terdeteksi)
# ============================================================
check_camera() {
    if [ -e /dev/video0 ]; then
        print_success "Camera terdeteksi: /dev/video0"
    else
        print_warning "/dev/video0 tidak ditemukan!"
        print_info "Pastikan CSI camera terpasang dengan benar."
        read -p "Lanjutkan tanpa camera? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            exit 0
        fi
    fi
}

# ============================================================
# Cek docker-compose.yml
# ============================================================
check_compose_file() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "docker-compose.yml tidak ditemukan: $COMPOSE_FILE"
        exit 1
    fi
}

check_environment_file() {
    if [ ! -f "$ENV_FILE" ]; then
        print_error "File deployment .env belum ada: $ENV_FILE"
        print_info "Salin .env.example ke .env, lalu isi semua nilai wajib."
        exit 1
    fi

    local required_key
    for required_key in FLASK_SECRET_KEY HAFIBOT_USERNAME HAFIBOT_PASSWORD; do
        if ! grep -Eq "^${required_key}=.+" "$ENV_FILE"; then
            print_error "Nilai wajib ${required_key} belum diisi di .env"
            exit 1
        fi
    done
}

compose() {
    sudo docker-compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

# ============================================================
# Start Docker Compose
# ============================================================
start_docker() {
    local build_flag=""
    if [ "$1" = "--build" ]; then
        build_flag="--build"
        print_info "Mode: Build ulang image lalu start..."
    else
        print_info "Mode: Start container..."
    fi

    # 1. Release the CSI camera before touching nvargus-daemon. Previously a
    # running container could keep an Argus client alive during --build, which
    # left the recreated pipeline vulnerable to CONSUMER TIMEOUT errors.
    print_info "Melepas camera client dari container lama..."
    compose down --remove-orphans || true
    sleep 1

    # 2. Restart nvargus-daemon only after all camera clients have stopped.
    restart_nvargus

    # 3. Cek camera
    check_camera

    # 4. Pastikan DISPLAY sudah di-set
    export DISPLAY=:0

    # 5. Jalankan xhost +local: (jaga-jaga kalau belum di-set via systemd)
    if command -v xhost &> /dev/null; then
        xhost +local: 2>/dev/null || true
    fi

    # 6. Start Docker Compose
    echo ""
    print_info "Menjalankan Docker Compose..."
    compose up -d $build_flag

    echo ""
    print_success "Container berhasil dijalankan!"
    print_info "Lihat logs dengan: ./scripts/start_docker.sh --logs"
    print_info "Stop dengan: ./scripts/start_docker.sh --stop"
}

# ============================================================
# Stop Docker Compose
# ============================================================
stop_docker() {
    print_info "Menghentikan container..."
    compose down
    print_success "Container berhasil dihentikan."
}

# ============================================================
# Restart Docker Compose (stop + restart nvargus + start)
# ============================================================
restart_docker() {
    print_info "Merestart container..."
    start_docker
}

# ============================================================
# Tampilkan logs
# ============================================================
show_logs() {
    print_info "Menampilkan logs container (Ctrl+C untuk keluar)..."
    echo ""
    compose logs -f --tail=100
}

# ============================================================
# Status
# ============================================================
show_status() {
    print_info "Status container:"
    echo ""
    compose ps

    echo ""
    print_info "Status nvargus-daemon:"
    if systemctl is-active --quiet nvargus-daemon 2>/dev/null; then
        print_success "nvargus-daemon: AKTIF"
    else
        print_warning "nvargus-daemon: TIDAK AKTIF"
    fi

    echo ""
    print_info "Camera device:"
    if [ -e /dev/video0 ]; then
        print_success "/dev/video0: Terdeteksi"
    else
        print_warning "/dev/video0: Tidak ditemukan"
    fi
}

# ============================================================
# Main
# ============================================================
print_header
check_compose_file
if [ "${1}" != "--help" ] && [ "${1}" != "-h" ]; then
    check_environment_file
fi

case "${1}" in
    --build|-b)
        start_docker --build
        ;;
    --stop|-s)
        stop_docker
        ;;
    --restart|-r)
        restart_docker
        ;;
    --logs|-l)
        show_logs
        ;;
    --status)
        show_status
        ;;
    --help|-h)
        echo "Penggunaan: $0 [opsi]"
        echo ""
        echo "Opsi:"
        echo "  (tanpa opsi)      Start container (restart nvargus + docker up)"
        echo "  --build, -b       Build ulang image lalu start"
        echo "  --stop, -s        Stop container"
        echo "  --restart, -r     Restart container (stop + restart nvargus + start)"
        echo "  --logs, -l        Lihat logs container (live)"
        echo "  --status          Cek status container, nvargus, dan camera"
        echo "  --help, -h        Tampilkan bantuan"
        ;;
    *)
        start_docker
        ;;
esac
