import os, pathlib, requests
from datetime import datetime, timezone

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS","NVDA").split(",") if s.strip()]
LOG_DIR = "logs"

def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def send_telegram(text: str):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    MAX = 3500
    chunks = [text[i:i+MAX] for i in range(0, len(text), MAX)]

    for c in chunks:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": c, "disable_web_page_preview": True},
            timeout=20
        )
        if r.status_code != 200:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")

def main():
    pathlib.Path(LOG_DIR).mkdir(exist_ok=True)
    ts = utc_now_iso()

    # Temporary scaffold summary so you can validate branch/workflow isolation.
    # Next step: replace this with real intraday data + 2H/4H logic.
    summary = "\n".join([
        f"[INTRADAY] Market Scan — {ts} UTC",
        "========================================",
        "This is the intraday branch pipeline ✅",
        f"Symbols: {', '.join(SYMBOLS)}",
        "",
        "Next: plug in intraday candles → resample 2H/4H → GPT analysis → Telegram.",
    ])

    (pathlib.Path(LOG_DIR) / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    send_telegram(summary)

if __name__ == "__main__":
    main()
