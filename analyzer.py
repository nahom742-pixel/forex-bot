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
        self.alpha_key = alpha_key or "demo"
        self.min_signal_strength = 70

    def get_price_data(self, pair: str) -> pd.DataFrame:
        """Fetch OHLCV data from Alpha Vantage free API."""
        try:
            # Convert pair format: EURUSD=X -> EUR/USD
            symbol = pair.replace("=X", "")
            from_currency = symbol[:3]
            to_currency = symbol[3:]

            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "FX_INTRADAY",
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "interval": "60min",
                "outputsize": "compact",
                "apikey": self.alpha_key
            }
            response = requests.get(url, params=params, timeout=15)
            data = response.json()

            key = "Time Series FX (60min)"
            if key not in data:
                # Try daily as fallback
                params["function"] = "FX_DAILY"
                response = requests.get(url, params=params, timeout=15)
                data = response.json()
                key = "Time Series FX (Daily)"

            if key not in data:
                logger.warning(f"No data for {pair} from Alpha Vantage")
                return None

            records = []
            for timestamp, values in data[key].items():
                records.append({
                    "Date": timestamp,
                    "Open": float(values["1. open"]),
                    "High": float(values["2. high"]),
                    "Low": float(values["3. low"]),
                    "Close": float(values["4. close"]),
                    "Volume": 0
                })

            df = pd.DataFrame(records)
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate technical indicators."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        macd_hist = macd - signal_line

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + (2 * std20)
        bb_lower = sma20 - (2 * std20)

        # Moving Averages
        ma50 = close.rolling(min(50, len(close))).mean()
        ma200 = close.rolling(min(200, len(close))).mean()

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # Stochastic
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(3).mean()

        # Pivot points
        prev_high = high.iloc[-2] if len(high) > 1 else high.iloc[-1]
        prev_low = low.iloc[-2] if len(low) > 1 else low.iloc[-1]
        prev_close = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
        pivot = (prev_high + prev_low + prev_close) / 3

        n = min(24, len(close))
        return {
            "current_price": round(close.iloc[-1], 5),
            "price_1h_ago": round(close.iloc[-2], 5) if len(close) > 1 else None,
            "price_24h_ago": round(close.iloc[-n], 5) if len(close) >= n else None,
            "rsi": round(rsi.iloc[-1], 2),
            "rsi_prev": round(rsi.iloc[-2], 2) if len(rsi) > 1 else None,
            "macd": round(macd.iloc[-1], 6),
            "macd_signal": round(signal_line.iloc[-1], 6),
            "macd_hist": round(macd_hist.iloc[-1], 6),
            "macd_hist_prev": round(macd_hist.iloc[-2], 6) if len(macd_hist) > 1 else None,
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
        """Send indicator data to AI for analysis."""
        pair_name = pair.replace("=X", "")

        prompt = f"""Je bent een professionele forex trader en analist. Analyseer de volgende technische indicatoren voor {pair_name} en geef een handelssignaal.

TECHNISCHE DATA ({pair_name}):
- Huidige prijs: {indicators['current_price']}
- Prijs 1u geleden: {indicators['price_1h_ago']}
- Prijs 24u geleden: {indicators['price_24h_ago']}
- 24u Hoog/Laag: {indicators['high_24h']} / {indicators['low_24h']}

MOMENTUM INDICATOREN:
- RSI (14): {indicators['rsi']} (vorige: {indicators['rsi_prev']})
- MACD: {indicators['macd']} | Signaal: {indicators['macd_signal']} | Histogram: {indicators['macd_hist']}
- Stochastic K/D: {indicators['stoch_k']} / {indicators['stoch_d']}

TREND INDICATOREN:
- MA50: {indicators['ma50']}
- MA200: {indicators['ma200']}
- Bollinger Boven/Mid/Onder: {indicators['bb_upper']} / {indicators['bb_mid']} / {indicators['bb_lower']}

VOLATILITEIT & NIVEAUS:
- ATR: {indicators['atr']}
- Pivot: {indicators['pivot']}
- Resistance 1: {indicators['resistance1']}
- Support 1: {indicators['support1']}

Geef je analyse in het volgende JSON formaat (alleen JSON, geen extra tekst):
{{
  "signal": "BUY" of "SELL" of "NEUTRAL",
  "strength": <getal 0-100>,
  "entry_price": <prijs>,
  "stop_loss": <prijs>,
  "take_profit": <prijs>,
  "risk_reward": <getal>,
  "timeframe": "korte termijn (1-4u)" of "middellange termijn (4-24u)",
  "main_reason": "<één zin hoofdreden in het Nederlands>",
  "analysis": "<gedetailleerde analyse in het Nederlands, 3-5 zinnen>",
  "confidence": "LAAG" of "GEMIDDELD" of "HOOG",
  "warnings": "<eventuele risico's>"
}}"""

        try:
            import json
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Je bent een professionele forex trader. Geef altijd alleen geldige JSON terug, zonder extra tekst of markdown."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3,
            )
            response_text = completion.choices[0].message.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            return json.loads(response_text.strip())
        except Exception as e:
            logger.error(f"AI analysis error for {pair}: {e}")
            return None

    def format_signal_message(self, pair: str, indicators: dict, analysis: dict) -> str:
        """Format the signal as a Telegram message."""
        pair_name = pair.replace("=X", "")
        signal = analysis["signal"]
        emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
        confidence_emoji = {"LAAG": "🟡", "GEMIDDELD": "🟠", "HOOG": "🟢"}.get(analysis["confidence"], "🟡")

        msg = (
            f"{emoji} *{signal} SIGNAAL — {pair_name}*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *Entry:* `{analysis['entry_price']}`\n"
            f"🛑 *Stop Loss:* `{analysis['stop_loss']}`\n"
            f"🎯 *Take Profit:* `{analysis['take_profit']}`\n"
            f"⚖️ *Risk/Reward:* {analysis['risk_reward']}:1\n"
            f"⏱ *Tijdframe:* {analysis['timeframe']}\n"
            f"💪 *Signaalsterkte:* {analysis['strength']}/100\n"
            f"{confidence_emoji} *Betrouwbaarheid:* {analysis['confidence']}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📋 *Reden:* {analysis['main_reason']}\n\n"
            f"🔍 *Analyse:*\n{analysis['analysis']}\n"
        )
        if analysis.get("warnings"):
            msg += f"\n⚠️ *Let op:* {analysis['warnings']}\n"

        msg += (
            f"\n━━━━━━━━━━━━━━━━━━\n"
            f"📊 RSI: {indicators['rsi']} | MACD: {'↑' if indicators['macd_hist'] > 0 else '↓'} "
            f"| Stoch: {indicators['stoch_k']:.0f}\n"
            f"🕐 {datetime.utcnow().strftime('%d/%m %H:%M')} UTC\n"
            f"⚠️ _Geen financieel advies. Trade op eigen risico._"
        )
        return msg

    async def analyze_all_pairs(self, pairs: list) -> list:
        """Analyze all pairs and return signal messages for strong signals."""
        signals = []
        for pair in pairs:
            try:
                logger.info(f"Analyzing {pair}...")
                df = self.get_price_data(pair)
                if df is None or len(df) < 30:
                    continue

                indicators = self.calculate_indicators(df)
                analysis = self.analyze_with_ai(pair, indicators)

                if analysis is None:
                    continue

                if analysis["signal"] != "NEUTRAL" and analysis["strength"] >= self.min_signal_strength:
                    msg = self.format_signal_message(pair, indicators, analysis)
                    signals.append(msg)
                    logger.info(f"Signal found for {pair}: {analysis['signal']} ({analysis['strength']})")
                else:
                    logger.info(f"No strong signal for {pair}: {analysis['signal']} ({analysis.get('strength', 0)})")

            except Exception as e:
                logger.error(f"Error analyzing {pair}: {e}")
                continue

        return signals
