#!/usr/bin/env bash
# =============================================================================
# start_edge_agent.sh - Resilient Boot Starter for Hafibot Jetson Edge Agent
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

SERVER_URL="${1:-http://100.86.76.95:5000}"
SERVER_HOST="$(echo "$SERVER_URL" | sed -E 's|https?://([^:/]+).*|\1|')"

echo "============================================================"
echo " [HAFIBOT] Preparing Edge Agent Boot Environment"
echo "============================================================"

# 1. Restart nvargus-daemon to clear any hung camera sessions
echo "[1/3] Ensuring clean CSI camera pipeline (nvargus-daemon)..."
if sudo -n systemctl restart nvargus-daemon 2>/dev/null; then
    echo "  nvargus-daemon restarted successfully."
elif [ "$(id -u)" -eq 0 ]; then
    systemctl restart nvargus-daemon || true
    echo "  nvargus-daemon restarted as root."
else
    echo "  [INFO] Skipping daemon restart or handled by systemd ExecStartPre."
fi
sleep 2

# 2. Wait for Wi-Fi / network connection
echo "[2/3] Waiting for network / Wi-Fi route to $SERVER_HOST..."
MAX_WAIT=45
WAITED=0
while ! ping -c 1 -W 2 "$SERVER_HOST" &>/dev/null; do
    if ping -c 1 -W 2 8.8.8.8 &>/dev/null; then
        echo "  Wi-Fi connected. Server route reachable or resolving via Tailscale/LAN."
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "  [WARNING] Network wait reached ${MAX_WAIT}s. Proceeding with launch..."
        break
    fi
    echo "  Waiting for Wi-Fi connectivity... (${WAITED}s/${MAX_WAIT}s)"
done

# 3. Launch Edge Agent
echo "[3/3] Starting Jetson Edge Agent (Server: $SERVER_URL)..."
exec python3 scripts/jetson_edge_agent.py \
    --server "$SERVER_URL" \
    --enable-voice \
    --speaker-volume 1.3
