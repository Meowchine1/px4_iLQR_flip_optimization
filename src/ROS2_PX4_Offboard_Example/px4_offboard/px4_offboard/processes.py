#!/usr/bin/env python3

import subprocess
import signal
import os
import sys
import time

# Список процессов gnome-terminal
gnome_terminals = []

# Команды для запуска
commands = [
    "MicroXRCEAgent udp4 -p 8888 ",
    "cd ~/PX4-Autopilot && make px4_sitl gz_x500" 
]

def terminate_processes(signal_received=None, frame=None):
    """Завершает все gnome-terminal и их дочерние процессы."""
    print("\n[INFO] Завершаем все процессы...")

    for pid in gnome_terminals:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)  # Отправляем SIGTERM всей группе
            time.sleep(2)
            os.killpg(os.getpgid(pid), signal.SIGKILL)  # Если не завершились, принудительно SIGKILL
        except ProcessLookupError:
            pass  # Если процесс уже завершен

    print("[INFO] Все процессы завершены.")
    sys.exit(0)

# Перехватываем SIGINT (Ctrl + C)
signal.signal(signal.SIGINT, terminate_processes)

# Запускаем каждый процесс в новом gnome-terminal
for command in commands:
    proc = subprocess.Popen(["gnome-terminal", "--", "bash", "-c", command], preexec_fn=os.setsid)
    gnome_terminals.append(proc.pid)
    time.sleep(1)

# Ждем завершения (не завершается, пока не нажмешь Ctrl + C)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    terminate_processes()
