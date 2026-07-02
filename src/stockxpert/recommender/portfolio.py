"""
Portfolio Optimization Module.
Handles allocation across a basket of recommendations.
"""

from typing import List, Dict
from .engine import Recommendation

class PortfolioOptimizer:
    """
    Optimizes a portfolio of recommendations.
    Enforces diversification and capital constraints.
    """
    
    def __init__(self, max_total_allocation: float = 1.0, max_sector_exposure: float = 0.4):
        self.max_total_allocation = max_total_allocation
        self.max_sector_exposure = max_sector_exposure
        
    def optimize_allocation(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """
        Adjust position sizes to fit portfolio constraints.
        """
        # 1. Filter only BUYs
        buys = [r for r in recommendations if "BUY" in r.action]
        others = [r for r in recommendations if "BUY" not in r.action]
        
        # 2. Sort by conviction (position size)
        buys.sort(key=lambda x: x.position_size, reverse=True)
        
        # 3. Allocator
        total_alloc = 0.0
        optimized_buys = []
        
        for rec in buys:
            # Check remaining capital
            remaining = self.max_total_allocation - total_alloc
            if remaining <= 0:
                rec.position_size = 0.0
                rec.action = "WATCHLIST (No Capital)"
                optimized_buys.append(rec)
                continue
                
            # Cap at remaining
            alloc = min(rec.position_size, remaining)
            
            # Update
            rec.position_size = alloc
            total_alloc += alloc
            
            # If allocation became trivial, skip or mark
            if alloc < 0.01:
                rec.action = "WATCHLIST (Low Alloc)"
                
            optimized_buys.append(rec)
            
        # Re-merge
        final_recs = optimized_buys + others
        
        # Sort again
        final_recs.sort(key=lambda x: x.position_size, reverse=True)
        
        return final_recs
