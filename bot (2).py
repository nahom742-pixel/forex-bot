import os
import asyncio
import logging
from datetime import datetime, time

from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update

from analyzer import ForexAnalyzer
from scheduler import TradingScheduler

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("telegram_token")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("telegram_chat_id")
GROQ_KEY = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")

analyzer = ForexAnalyzer(GROQ_KEY)
scheduler = TradingScheduler()

MAJOR_PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
    "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X"
]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Forex AI Bot actief!*\n\n"
        "Ik monitor 24/7 alle major forex paren en stuur je buy/sell signalen.\n\n"
        "Commando's:\n"
        "/start - Bot starten\n"
        "/status - Huidige marktstatus\n"
        "/scan - Directe marktscan uitvoeren\n"
        "/pairs - Welke paren ik monitor",
        parse_mode="Markdown"
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = scheduler.get_current_session()
    pairs_text = "\n".join([f"• {p.replace('=X', '')}" for p in MAJOR_PAIRS])
    await update.message.reply_text(
        f"📊 *Bot Status*\n\n"
        f"✅ Bot actief\n"
        f"🕐 Tijd (UTC): {datetime.utcnow().strftime('%H:%M')}\n"
        f"📍 Trading sessie: {session}\n"
        f"💱 Monitored paren:\n{pairs_text}",
        parse_mode="Markdown"
    )

async def manual_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Bezig met scannen... even geduld.")
    signals = await analyzer.analyze_all_pairs(MAJOR_PAIRS)
    if signals:
        for signal in signals:
            await update.message.reply_text(signal, parse_mode="Markdown")
    else:
        await update.message.reply_text("📭 Geen sterke signalen gevonden op dit moment.")

async def pairs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pairs_text = "\n".join([f"• {p.replace('=X', '')}" for p in MAJOR_PAIRS])
    await update.message.reply_text(f"💱 *Gemonitorde paren:*\n\n{pairs_text}", parse_mode="Markdown")

async def scheduled_scan(app: Application):
    """Runs periodically to scan the market and send signals."""
    logger.info("Running scheduled market scan...")
    try:
        signals = await analyzer.analyze_all_pairs(MAJOR_PAIRS)
        if signals:
            for signal in signals:
                await app.bot.send_message(
                    chat_id=CHAT_ID,
                    text=signal,
                    parse_mode="Markdown"
                )
                await asyncio.sleep(1)
        else:
            logger.info("No strong signals found in this scan.")
    except Exception as e:
        logger.error(f"Error during scheduled scan: {e}")

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("scan", manual_scan))
    app.add_handler(CommandHandler("pairs", pairs))

    # Schedule market scans based on trading sessions
    interval_minutes = scheduler.get_scan_interval()
    app.job_queue.run_repeating(
        lambda ctx: asyncio.create_task(scheduled_scan(app)),
        interval=interval_minutes * 60,
        first=30  # First scan after 30 seconds
    )

    logger.info("🤖 Forex AI Bot started!")
    app.run_polling()

if __name__ == "__main__":
    main()
