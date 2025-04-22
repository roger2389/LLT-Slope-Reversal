import pandas as pd
import numpy as np
from typing import Tuple, Dict

class LLTTrendStrategy:
    def __init__(self, d: int = 30):
        self.d = d
        self.alpha = 2 / (d + 1)

    def cal_LLT(self, price: pd.Series) -> pd.Series:
        LLT = []
        price_value = price.values
        LLT.append(price_value[0])
        LLT.append(price_value[1])
        for i in range(2, len(price_value)):
            v = (self.alpha - self.alpha**2 / 4) * price_value[i] + \
                (self.alpha**2 / 2) * price_value[i - 1] - \
                (self.alpha - 3 * (self.alpha**2) / 4) * price_value[i - 2] + \
                2 * (1 - self.alpha) * LLT[i - 1] - (1 - self.alpha)**2 * LLT[i - 2]
            LLT.append(v)
        return pd.Series(LLT, index=price.index)

    def generate_signal(self, price: pd.Series) -> pd.Series:
        llt = self.cal_LLT(price)
        slope = llt.diff()
        signal = ((slope > 0) * 1 + (slope < 0) * -1).shift(1).fillna(0)
        return signal

    def backtest(self, price: pd.Series) -> pd.DataFrame:
        signal = self.generate_signal(price)
        ret = price.pct_change().fillna(0)
        strat_ret = signal * ret
        nav = (1 + strat_ret).cumprod()
        return pd.DataFrame({
            'price': price,
            'signal': signal,
            'strategy_return': strat_ret,
            'nav': nav
        })

# 搜尋最佳 d 值
def run_grid_search(price: pd.Series, d_range: range, split_date: str) -> pd.DataFrame:
    price_is = price[price.index < split_date]  # 樣本內
    price_oos = price[price.index >= split_date]  # 樣本外

    result = []
    for d in d_range:
        strategy = LLTTrendStrategy(d=d)

        # 樣本內回測，用來選參數
        df_is = strategy.backtest(price_is)
        nav_is = df_is['nav'].iloc[-1]
        ann_return_is = df_is['strategy_return'].mean() * 252
        sharpe_is = df_is['strategy_return'].mean() / df_is['strategy_return'].std() * np.sqrt(252)
        mdd_is = (df_is['nav'] / df_is['nav'].cummax() - 1).min()

        # 樣本外回測，用來驗證泛化
        df_oos = strategy.backtest(price_oos)
        nav_oos = df_oos['nav'].iloc[-1]
        ann_return_oos = df_oos['strategy_return'].mean() * 252
        sharpe_oos = df_oos['strategy_return'].mean() / df_oos['strategy_return'].std() * np.sqrt(252)
        mdd_oos = (df_oos['nav'] / df_oos['nav'].cummax() - 1).min()

        result.append({
            'd': d,
            'ann_return_is': ann_return_is,
            'sharpe_is': sharpe_is,
            'mdd_is': mdd_is,
            'ann_return_oos': ann_return_oos,
            'sharpe_oos': sharpe_oos,
            'mdd_oos': mdd_oos,
        })

    return pd.DataFrame(result).sort_values(by='sharpe_is', ascending=False)
