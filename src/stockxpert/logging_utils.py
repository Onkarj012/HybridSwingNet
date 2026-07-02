"""
Logging utilities for StockXpert.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    run_dir: Optional[Path] = None,
    log_file: str = "stockxpert.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging with both file and console handlers.
    
    Args:
        run_dir: Directory to save log file (if None, console only)
        log_file: Name of log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger("stockxpert")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if run_dir provided)
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(run_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
