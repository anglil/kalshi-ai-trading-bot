#!/bin/bash
# Kalshi AI Trading Bot - Persistent Runner with Auto-Restart
# Runs the bot in live trading mode and restarts it if it crashes

BOT_DIR="/home/ubuntu/kalshi-ai-trading-bot"
LOG_FILE="$BOT_DIR/logs/bot_runner.log"
PID_FILE="$BOT_DIR/bot.pid"

mkdir -p "$BOT_DIR/logs"

echo "[$(date)] Starting Kalshi AI Trading Bot (LIVE MODE)" | tee -a "$LOG_FILE"

cd "$BOT_DIR"

while true; do
    echo "[$(date)] Launching bot process..." | tee -a "$LOG_FILE"
    python3 beast_mode_bot.py --live >> "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    echo "[$(date)] Bot exited with code $EXIT_CODE. Restarting in 10 seconds..." | tee -a "$LOG_FILE"
    sleep 10
done
