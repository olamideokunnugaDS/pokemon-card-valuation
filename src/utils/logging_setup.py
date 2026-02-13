"""
Logging setup utility for consistent logging across all modules.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for terminal."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    experiment_name: Optional[str] = None,
    colored: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, only console logging)
        experiment_name: Name of experiment (added to log file name)
        colored: Use colored output for console logging
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("pokemon_valuation")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if colored:
        console_format = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if log file specified)
    if log_file:
        log_path = Path(log_file)
        
        # Add experiment name and timestamp to log file
        if experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_path.parent / f"{experiment_name}_{timestamp}_{log_path.name}"
        
        # Create directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"pokemon_valuation.{name}")


class ExperimentLogger:
    """Logger for tracking experiment progress and metrics."""
    
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to store logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logging(
            log_level="INFO",
            log_file=str(log_file),
            experiment_name=experiment_name
        )
    
    def log_hyperparameters(self, hyperparams: dict) -> None:
        """Log experiment hyperparameters."""
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("=" * 80)
        self.logger.info("Hyperparameters:")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 80)
    
    def log_metrics(self, metrics: dict, epoch: Optional[int] = None, phase: str = "train") -> None:
        """Log training/validation metrics."""
        prefix = f"Epoch {epoch} - {phase.capitalize()}" if epoch is not None else phase.capitalize()
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{prefix} - {metric_str}")
    
    def log_milestone(self, message: str) -> None:
        """Log a significant milestone or event."""
        self.logger.info("=" * 80)
        self.logger.info(f"MILESTONE: {message}")
        self.logger.info("=" * 80)


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(
        log_level="DEBUG",
        log_file="logs/test.log",
        experiment_name="test_experiment"
    )
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test experiment logger
    exp_logger = ExperimentLogger("test_experiment")
    exp_logger.log_hyperparameters({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    })
    
    exp_logger.log_metrics({
        "loss": 0.523,
        "accuracy": 0.876
    }, epoch=1, phase="train")
    
    exp_logger.log_milestone("Training completed successfully!")
