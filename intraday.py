import os
import requests
from datetime import datetime, timezone

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS","NVDA").split(",") if s.strip()]

def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def main():
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    msg = "\n".join([
        f"[INTRADAY] Test message — {utc_now_iso()} UTC",
        f"Symbols: {', '.join(SYMBOLS)}",
        "Workflow_dispatch is working ✅"
    ])

    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": msg},
        timeout=20
    )
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")

    print("Sent Telegram test message.")

if __name__ == "__main__":
    main()
