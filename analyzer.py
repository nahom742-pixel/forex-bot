import os
import requests
import pandas as pd
import numpy as np
from groq import Groq
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ForexAnalyzer:
    def __init__(self, api_key: str, alpha_key: str = None):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.min_signal_strength = 70
        self.twelve_key = os.getenv("TWELVE_DATA_KEY", "demo")

    def get_price_data(self, pair: str) -> pd.DataFrame:
        """Fetch OHLCV data from Twelve Data free API."""
        try:
            symbol = pair.replace("=X", "")
            forex_symbol = f"{symbol[:3]}/{symbol[3:]}"

            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": forex_symbol,
                "interval": "1h",
                "outputsize": 100,
                "apikey": self.twelve_key
            }
            response = requests.get(url, params=params, timeout=15)
            data = response.json()

            if "values" not in data:
                logger.warning(f"No data for {pair}: {data.get('message', 'unknown error')}")
                return None

            records = []
            for item in data["values"]:
                records.append({
                    "Date": item["datetime"],
                    "Open": float(item["open"]),
                    "High": float(item["high"]),
                    "Low": float(item["low"]),
                    "Close": float(item["close"]),
                })

            df = pd.DataFrame(records)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        macd_hist = macd - signal_line

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + (2 * std20)
        bb_lower = sma20 - (2 * std20)

        ma50 = close.rolling(min(50, len(close))).mean()
        ma200 = close.rolling(min(200, len(close))).mean()

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(3).mean()

        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        prev_close = close.iloc[-2]
        pivot = (prev_high + prev_low + prev_close) / 3

        n = min(24, len(close))
        return {
            "current_price": round(close.iloc[-1], 5),
            "price_1h_ago": round(close.iloc[-2], 5),
            "price_24h_ago": round(close.iloc[-n], 5),
            "rsi": round(rsi.iloc[-1], 2),
            "rsi_prev": round(rsi.iloc[-2], 2),
            "macd": round(macd.iloc[-1], 6),
            "macd_signal": round(signal_line.iloc[-1], 6),
            "macd_hist": round(macd_hist.iloc[-1], 6),
            "macd_hist_prev": round(macd_hist.iloc[-2], 6),
            "bb_upper": round(bb_upper.iloc[-1], 5),
            "bb_lower": round(bb_lower.iloc[-1], 5),
            "bb_mid": round(sma20.iloc[-1], 5),
            "ma50": round(ma50.iloc[-1], 5),
            "ma200": round(ma200.iloc[-1], 5),
            "atr": round(atr.iloc[-1], 5),
            "stoch_k": round(stoch_k.iloc[-1], 2),
            "stoch_d": round(stoch_d.iloc[-1], 2),
            "pivot": round(pivot, 5),
            "resistance1": round(2 * pivot - prev_low, 5),
            "support1": round(2 * pivot - prev_high, 5),
            "high_24h": round(high.iloc[-n:].max(), 5),
            "low_24h": round(low.iloc[-n:].min(), 5),
        }

    def analyze_with_ai(self, pair: str, indicators: dict) -> dict:
        pair_name = pair.replace("=X", "")
        prompt = f"""Je bent een professionele forex trader. Analyseer {pair_name}:

Prijs: {indicators['current_price']} | 1u geleden: {indicators['price_1h_ago']} | 24u geleden: {indicators['price_24h_ago']}
24u H/L: {indicators['high_24h']} / {indicators['low_24h']}
RSI: {indicators['rsi']} | MACD hist: {indicators['macd_hist']} | Stoch K/D: {indicators['stoch_k']}/{indicators['stoch_d']}
MA50: {indicators['ma50']} | MA200: {indicators['ma200']}
BB: {indicators['bb_upper']}/{indicators['bb_mid']}/{indicators['bb_lower']}
ATR: {indicators['atr']} | Pivot: {indicators['pivot']} | R1: {indicators['resistance1']} | S1: {indicators['support1']}

Geef alleen JSON:
{{"signal":"BUY/SELL/NEUTRAL","strength":0-100,"entry_price":0.0,"stop_loss":0.0,"take_profit":0.0,"risk_reward":0.0,"timeframe":"korte termijn (1-4u)","main_reason":"...","analysis":"...","confidence":"LAAG/GEMIDDELD/HOOG","warnings":"..."}}"""

        try:
            import json
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Forex analist. Alleen geldige JSON, geen markdown."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.3,
            )
            response_text = completion.choices[0].message.content.strip()
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            return json.loads(response_text.strip())
        except Exception as e:
            logger.error(f"AI error for {pair}: {e}")
            return None

    def format_signal_message(self, pair: str, indicators: dict, analysis: dict) -> str:
        pair_name = pair.replace("=X", "")
        signal = analysis["signal"]
        emoji = "🟢" if signal == "BUY" else "🔴"
        conf_emoji = {"LAAG": "🟡", "GEMIDDELD": "🟠", "HOOG": "🟢"}.get(analysis["confidence"], "🟡")

        msg = (
            f"{emoji} *{signal} — {pair_name}*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *Entry:* `{analysis['entry_price']}`\n"
            f"🛑 *Stop Loss:* `{analysis['stop_loss']}`\n"
            f"🎯 *Take Profit:* `{analysis['take_profit']}`\n"
            f"⚖️ *R/R:* {analysis['risk_reward']}:1\n"
            f"⏱ *Tijdframe:* {analysis['timeframe']}\n"
            f"💪 *Sterkte:* {analysis['strength']}/100\n"
            f"{conf_emoji} *Betrouwbaarheid:* {analysis['confidence']}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📋 *Reden:* {analysis['main_reason']}\n\n"
            f"🔍 *Analyse:* {analysis['analysis']}\n"
        )
        if analysis.get("warnings"):
            msg += f"\n⚠️ *Let op:* {analysis['warnings']}\n"
        msg += f"\n🕐 {datetime.utcnow().strftime('%d/%m %H:%M')} UTC\n_Geen financieel advies._"
        return msg

    async def analyze_all_pairs(self, pairs: list) -> list:
        signals = []
        for pair in pairs:
            try:
                logger.info(f"Analyzing {pair}...")
                df = self.get_price_data(pair)
                if df is None or len(df) < 30:
                    continue
                indicators = self.calculate_indicators(df)
                analysis = self.analyze_with_ai(pair, indicators)
                if analysis and analysis["signal"] != "NEUTRAL" and analysis["strength"] >= self.min_signal_strength:
                    signals.append(self.format_signal_message(pair, indicators, analysis))
                    logger.info(f"Signal: {pair} {analysis['signal']} ({analysis['strength']})")
                else:
                    logger.info(f"No signal for {pair}")
            except Exception as e:
                logger.error(f"Error {pair}: {e}")
        return signals
