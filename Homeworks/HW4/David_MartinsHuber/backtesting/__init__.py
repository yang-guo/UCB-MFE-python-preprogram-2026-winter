"""HW4 backtesting v1: an independent implementation informed by Lecture 8."""

from .contracts import DecisionContext, FitData, Phase, Portfolio, Strategy
from .engine import Backtester, BacktestResult, HeldPriceGapError
from .famafrench import FamaFrench5, FactorResult
from .market_data import MarketData, MarketView
from .oms import OMS
from .analytics import BacktestAnalytics
from .risk import (RiskManager, RiskPolicy, RiskContext, RiskAdjustment, RiskDecision,
                   SingleNameCap, DrawdownGate, VolatilityTarget)
from .portfolio import (AllocationContext, EqualWeightOptimizer, MinimumVarianceOptimizer,
                        MaximumSharpeOptimizer, MaximumDiversificationOptimizer,
                        EqualRiskContributionOptimizer, ERCOptimizer, MSPOptimizer, MDPOptimizer,
                        ledoit_wolf_covariance, PortfolioOptimizationError, PortfolioOptimizer,
                        PortfolioStrategy, ShadowSpec, ShadowPortfolioBacktester,
                        PortfolioBacktestResult)

__all__ = ["Backtester", "BacktestResult", "BacktestAnalytics", "DecisionContext", "FamaFrench5",
           "FactorResult", "FitData",
           "HeldPriceGapError", "MarketData", "MarketView", "OMS", "Phase", "Portfolio", "Strategy",
           "AllocationContext", "EqualWeightOptimizer", "MinimumVarianceOptimizer",
           "MaximumSharpeOptimizer", "MaximumDiversificationOptimizer",
           "EqualRiskContributionOptimizer", "ERCOptimizer", "MSPOptimizer", "MDPOptimizer",
           "ledoit_wolf_covariance", "PortfolioOptimizationError",
           "PortfolioOptimizer", "PortfolioStrategy", "ShadowSpec",
           "ShadowPortfolioBacktester", "PortfolioBacktestResult",
           "RiskManager", "RiskPolicy", "RiskContext", "RiskAdjustment", "RiskDecision",
           "SingleNameCap", "DrawdownGate", "VolatilityTarget"]
