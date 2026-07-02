"""V3 horizon-specialized model exports.

For V3, the main supported architecture is ``StockXpertModelV3``. These names
are kept as aliases to make migration from V1 explicit without retaining stock
ID embeddings in the V3 namespace.
"""

from stockxpert.v3.models.stockxpert import StockXpertModelV3

BaseHorizonModel = StockXpertModelV3
ShortTermModel = StockXpertModelV3
MediumTermModel = StockXpertModelV3
LongTermModel = StockXpertModelV3

__all__ = ["BaseHorizonModel", "ShortTermModel", "MediumTermModel", "LongTermModel"]

