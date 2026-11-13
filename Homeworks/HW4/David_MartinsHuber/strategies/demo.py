"""A small interface demonstration, not our final researched trading model."""

from backtesting import Strategy


class MonthlyMomentum(Strategy):
    def __init__(self, strategy_id="momentum_demo", lookback=63, top_n=10):
        super().__init__(strategy_id)
        if not isinstance(lookback, int) or lookback < 1 or not isinstance(top_n, int) or top_n < 1:
            raise ValueError("lookback and top_n must be positive integers")
        self.lookback, self.top_n = lookback, top_n

    def propose_buys(self, context):
        if context.signal_date.month == context.execution_date.month:
            return {}
        window = context.prices.tail(self.lookback + 1)
        if len(window) <= self.lookback:
            return {}
        allowed = set(context.membership.symbol)
        symbols = [s for s in sorted(allowed) if s in window and window[s].notna().all()]
        scores = (window[symbols].iloc[-1] / window[symbols].iloc[0] - 1).sort_values(
            ascending=False, kind="stable")
        selected = scores.head(self.top_n).index
        budget = min(context.portfolio.available_cash, context.portfolio.nav / 3)
        return {symbol: budget / len(selected) for symbol in selected} if len(selected) and budget > 0 else {}
