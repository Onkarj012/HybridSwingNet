"""
Feature schemas and validation.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class FeatureSchema:
    """Schema defining feature groups for multi-scale tensors."""
    
    short_features: List[str]
    mid_features: List[str]
    long_features: List[str]
    context_features: List[str]
    
    def validate(self) -> None:
        """Validate that all feature lists are non-empty and unique."""
        if not self.short_features:
            raise ValueError("short_features cannot be empty")
        if not self.mid_features:
            raise ValueError("mid_features cannot be empty")
        if not self.long_features:
            raise ValueError("long_features cannot be empty")
        if not self.context_features:
            raise ValueError("context_features cannot be empty")
        
        # Check for duplicates within each group
        for name, features in [
            ('short', self.short_features),
            ('mid', self.mid_features),
            ('long', self.long_features),
            ('context', self.context_features)
        ]:
            if len(features) != len(set(features)):
                raise ValueError(f"{name}_features contains duplicates")
    
    def get_all_features(self) -> List[str]:
        """Get set of all unique features across all groups."""
        all_features = set()
        all_features.update(self.short_features)
        all_features.update(self.mid_features)
        all_features.update(self.long_features)
        all_features.update(self.context_features)
        return sorted(list(all_features))
