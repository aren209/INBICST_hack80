#!/usr/bin/env bash
# Automated installation script for a freshly installed Debian 12 / Ubuntu 22.04+ server.
# Installs base utilities, Docker Engine + Docker Compose v2, NVIDIA drivers (if GPU present & missing),
# NVIDIA Container Toolkit, and launches the local RAG Navigator stack.

set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Запустите скрипт с правами root: sudo ./scripts/setup_debian_server.sh"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive

echo "[1/5] Обновление пакетной базы и установка базовых утилит..."
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  git \
  pciutils

echo "[2/5] Установка Docker Engine и плагина Docker Compose v2..."
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  . /etc/os-release
  DISTRO_ID="${ID:-debian}"
  curl -fsSL "https://download.docker.com/linux/${DISTRO_ID}/gpg" | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${DISTRO_ID} ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y --no-install-recommends \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
  systemctl enable --now docker
else
  echo "Docker уже установлен: $(docker --version)"
fi

echo "[3/5] Проверка GPU NVIDIA и установка NVIDIA Container Toolkit..."
if lspci | grep -qi nvidia; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Обнаружена видеокарта NVIDIA без драйвера. Устанавливаю драйвер..."
    . /etc/os-release
    if [[ "${ID:-}" == "ubuntu" ]]; then
      apt-get install -y ubuntu-drivers-common
      ubuntu-drivers install --gpgpu
    else
      sed -i 's/main$/main contrib non-free non-free-firmware/' /etc/apt/sources.list || true
      apt-get update -y
      apt-get install -y "linux-headers-$(uname -r)" nvidia-driver firmware-misc-nonfree
    fi
    echo "ВНИМАНИЕ: Драйвер NVIDIA установлен. Перезагрузите сервер (sudo reboot) и запустите скрипт повторно!"
    exit 0
  fi

  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
  fi
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "ПРЕДУПРЕЖДЕНИЕ: GPU NVIDIA не обнаружен (lspci)."
fi

echo "[4/5] Подготовка конфигурации .env..."
cd "$PROJECT_DIR"
if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo "[5/5] Сборка и запуск контейнеров Docker Compose (qdrant + vllm-qwen38 + rag-app)..."
docker compose up -d --build

echo ""
echo "Установка завершена!"
echo "  • Веб-интерфейс и API: http://localhost:8000"
echo "  • Swagger UI:          http://localhost:8000/docs"
echo "  • Статус контейнеров:  docker compose ps"
echo "  • Логи инициализации:  docker compose logs -f"
