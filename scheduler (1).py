from datetime import datetime, timezone


class TradingScheduler:
    """
    Determines the current forex trading session and optimal scan intervals.
    Sessions (UTC):
      - Tokyo:    00:00 - 09:00
      - London:   08:00 - 17:00
      - New York: 13:00 - 22:00
      - Overlap London/NY: 13:00 - 17:00  ← best liquidity
      - Dead zone: 22:00 - 00:00
    """

    SESSIONS = {
        "🗼 Tokyo": (0, 9),
        "🇬🇧 London": (8, 17),
        "🗽 New York": (13, 22),
        "😴 Rustige periode": (22, 24),
    }

    def get_current_hour(self) -> int:
        return datetime.now(timezone.utc).hour

    def get_current_session(self) -> str:
        hour = self.get_current_hour()

        # Best overlap: London + New York
        if 13 <= hour < 17:
            return "🔥 London + New York overlap (beste liquiditeit)"
        elif 8 <= hour < 13:
            return "🇬🇧 London sessie"
        elif 13 <= hour < 22:
            return "🗽 New York sessie"
        elif 0 <= hour < 9:
            return "🗼 Tokyo sessie"
        else:
            return "😴 Rustige periode (22:00-00:00 UTC)"

    def get_scan_interval(self) -> int:
        """
        Return scan interval in minutes based on current session.
        - Overlap (best): every 15 min
        - London / NY:    every 20 min
        - Tokyo:          every 30 min
        - Dead zone:      every 60 min
        """
        hour = self.get_current_hour()

        if 13 <= hour < 17:   # London/NY overlap
            return 15
        elif 8 <= hour < 22:  # London or NY
            return 20
        elif 0 <= hour < 9:   # Tokyo
            return 30
        else:                  # Dead zone
            return 60

    def is_prime_time(self) -> bool:
        """Returns True during high-liquidity sessions."""
        hour = self.get_current_hour()
        return 8 <= hour < 22
