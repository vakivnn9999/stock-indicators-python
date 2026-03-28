"""
Entry Signal Functions
~~~~~~~~~~~~~~~~~~~~~~

Contains functions for determining entry signals based on
technical indicators including Stochastic Oscillator, SMA, and MACD.
"""

from typing import Iterable, Optional

from stock_indicators import indicators
from stock_indicators.indicators.common.quote import Quote

# Minimum number of quotes required by the slowest indicator per timeframe.
# MACD(12, 26, 9) on 15m needs at least 26 periods; SMA(10) on 1h needs at least 10 periods.
_MIN_QUOTES_15M = 26
_MIN_QUOTES_1H = 10


def _check_stoch_crossup_buy(
    quotes_15m: Iterable[Quote],
    quotes_1h: Iterable[Quote],
) -> bool:
    """Check Stochastic cross-up buy signal with trend confirmation filters.

    Returns True when Stochastic %K crosses above %D (buy signal) AND
    all of the following trend confirmation conditions are met using the & operator:
      - sma10_15m > sma25_15m  (short-term trend above medium-term on 15m)
      - sma20_15m > sma25_15m  (medium-term trend above slower medium-term on 15m)
      - macd_signal > 0        (MACD signal line is positive)
      - sma5_1h > sma10_1h     (fast SMA above slow SMA on 1h, uptrend)

    Parameters:
        `quotes_15m` : Iterable[Quote]
            Historical price quotes for the 15-minute timeframe.

        `quotes_1h` : Iterable[Quote]
            Historical price quotes for the 1-hour timeframe.

    Returns:
        `bool`
            True if Stochastic cross-up buy conditions are satisfied, False otherwise.
    """
    quotes_15m = list(quotes_15m)
    quotes_1h = list(quotes_1h)

    if len(quotes_15m) < _MIN_QUOTES_15M or len(quotes_1h) < _MIN_QUOTES_1H:
        return False

    # --- 15-minute indicators ---
    stoch_15m = indicators.get_stoch(quotes_15m)
    sma10_15m_results = indicators.get_sma(quotes_15m, 10)
    sma20_15m_results = indicators.get_sma(quotes_15m, 20)
    sma25_15m_results = indicators.get_sma(quotes_15m, 25)
    macd_15m_results = indicators.get_macd(quotes_15m)

    # --- 1-hour indicators ---
    sma5_1h_results = indicators.get_sma(quotes_1h, 5)
    sma10_1h_results = indicators.get_sma(quotes_1h, 10)

    # Get the latest values
    stoch_current = stoch_15m[-1]
    stoch_previous = stoch_15m[-2] if len(stoch_15m) >= 2 else None

    sma10_15m: Optional[float] = sma10_15m_results[-1].sma
    sma20_15m: Optional[float] = sma20_15m_results[-1].sma
    sma25_15m: Optional[float] = sma25_15m_results[-1].sma
    macd_signal: Optional[float] = macd_15m_results[-1].signal

    sma5_1h: Optional[float] = sma5_1h_results[-1].sma
    sma10_1h: Optional[float] = sma10_1h_results[-1].sma

    # Validate all required values are available
    if any(v is None for v in [
        stoch_current.oscillator, stoch_current.signal,
        sma10_15m, sma20_15m, sma25_15m, macd_signal,
        sma5_1h, sma10_1h,
    ]):
        return False

    if stoch_previous is None or stoch_previous.oscillator is None or stoch_previous.signal is None:
        return False

    # Stochastic cross-up: %K crosses above %D
    stoch_crossup = (
        stoch_previous.oscillator <= stoch_previous.signal
        and stoch_current.oscillator > stoch_current.signal
    )

    # Trend confirmation conditions combined with &
    trend_conditions = (
        (sma10_15m > sma25_15m)
        & (sma20_15m > sma25_15m)
        & (macd_signal > 0)
        & (sma5_1h > sma10_1h)
    )

    return bool(stoch_crossup and trend_conditions)


def _check_stoch_crossdown_sell(
    quotes_15m: Iterable[Quote],
    quotes_1h: Iterable[Quote],
) -> bool:
    """Check Stochastic cross-down sell signal with trend confirmation filters.

    Returns True when Stochastic %K crosses below %D (sell signal) AND
    all of the following trend confirmation conditions are met using the & operator
    (reverse of the buy conditions):
      - sma10_15m < sma25_15m  (short-term trend below medium-term on 15m)
      - sma20_15m < sma25_15m  (medium-term trend below slower medium-term on 15m)
      - macd_signal < 0        (MACD signal line is negative)
      - sma5_1h < sma10_1h     (fast SMA below slow SMA on 1h, downtrend)

    Parameters:
        `quotes_15m` : Iterable[Quote]
            Historical price quotes for the 15-minute timeframe.

        `quotes_1h` : Iterable[Quote]
            Historical price quotes for the 1-hour timeframe.

    Returns:
        `bool`
            True if Stochastic cross-down sell conditions are satisfied, False otherwise.
    """
    quotes_15m = list(quotes_15m)
    quotes_1h = list(quotes_1h)

    if len(quotes_15m) < _MIN_QUOTES_15M or len(quotes_1h) < _MIN_QUOTES_1H:
        return False

    # --- 15-minute indicators ---
    stoch_15m = indicators.get_stoch(quotes_15m)
    sma10_15m_results = indicators.get_sma(quotes_15m, 10)
    sma20_15m_results = indicators.get_sma(quotes_15m, 20)
    sma25_15m_results = indicators.get_sma(quotes_15m, 25)
    macd_15m_results = indicators.get_macd(quotes_15m)

    # --- 1-hour indicators ---
    sma5_1h_results = indicators.get_sma(quotes_1h, 5)
    sma10_1h_results = indicators.get_sma(quotes_1h, 10)

    # Get the latest values
    stoch_current = stoch_15m[-1]
    stoch_previous = stoch_15m[-2] if len(stoch_15m) >= 2 else None

    sma10_15m: Optional[float] = sma10_15m_results[-1].sma
    sma20_15m: Optional[float] = sma20_15m_results[-1].sma
    sma25_15m: Optional[float] = sma25_15m_results[-1].sma
    macd_signal: Optional[float] = macd_15m_results[-1].signal

    sma5_1h: Optional[float] = sma5_1h_results[-1].sma
    sma10_1h: Optional[float] = sma10_1h_results[-1].sma

    # Validate all required values are available
    if any(v is None for v in [
        stoch_current.oscillator, stoch_current.signal,
        sma10_15m, sma20_15m, sma25_15m, macd_signal,
        sma5_1h, sma10_1h,
    ]):
        return False

    if stoch_previous is None or stoch_previous.oscillator is None or stoch_previous.signal is None:
        return False

    # Stochastic cross-down: %K crosses below %D
    stoch_crossdown = (
        stoch_previous.oscillator >= stoch_previous.signal
        and stoch_current.oscillator < stoch_current.signal
    )

    # Trend confirmation conditions combined with & (reverse of buy conditions)
    trend_conditions = (
        (sma10_15m < sma25_15m)
        & (sma20_15m < sma25_15m)
        & (macd_signal < 0)
        & (sma5_1h < sma10_1h)
    )

    return bool(stoch_crossdown and trend_conditions)
