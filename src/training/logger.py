"""
Logging utilities for training
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path


class WandbLogger:
    """
    Weights & Biases logger
    
    Tracks metrics, hyperparameters, and artifacts
    """
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        enabled: bool = True
    ):
        """
        Initialize W&B logger
        
        Args:
            project: W&B project name
            name: Run name
            config: Config dict to log
            enabled: Enable logging
        """
        self.enabled = enabled
        
        if not enabled:
            print("W&B logging disabled")
            return
        
        try:
            import wandb
            self.wandb = wandb
            
            # Initialize run
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                resume='allow'
            )
            
            print(f"✓ W&B initialized: {project}/{name}")
        except ImportError:
            print("⚠ wandb not installed. Install with: pip install wandb")
            self.enabled = False
        except Exception as e:
            print(f"⚠ W&B initialization failed: {e}")
            self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics"""
        if not self.enabled:
            return
        
        self.wandb.log(metrics, step=step)
    
    def log_config(self, config: dict):
        """Log configuration"""
        if not self.enabled:
            return
        
        self.wandb.config.update(config)
    
    def finish(self):
        """Finish run"""
        if not self.enabled:
            return
        
        self.wandb.finish()


class TensorBoardLogger:
    """
    TensorBoard logger
    
    Alternative to W&B for local logging
    """
    
    def __init__(
        self,
        log_dir: str,
        enabled: bool = True
    ):
        """
        Initialize TensorBoard logger
        
        Args:
            log_dir: Directory for logs
            enabled: Enable logging
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        
        if not enabled:
            print("TensorBoard logging disabled")
            return
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            
            print(f"✓ TensorBoard initialized: {log_dir}")
        except ImportError:
            print("⚠ tensorboard not installed. Install with: pip install tensorboard")
            self.enabled = False
        except Exception as e:
            print(f"⚠ TensorBoard initialization failed: {e}")
            self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics"""
        if not self.enabled:
            return
        
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def close(self):
        """Close writer"""
        if not self.enabled:
            return
        
        self.writer.close()


class ConsoleLogger:
    """
    Simple console logger
    
    Prints metrics to stdout
    """
    
    def __init__(self, log_interval: int = 100):
        """
        Initialize console logger
        
        Args:
            log_interval: Print every N steps
        """
        self.log_interval = log_interval
    
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to console"""
        if step % self.log_interval == 0:
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"Step {step}: {metrics_str}")


class MultiLogger:
    """
    Combine multiple loggers
    
    Logs to all registered loggers simultaneously
    """
    
    def __init__(self, loggers: list):
        """
        Initialize multi-logger
        
        Args:
            loggers: List of logger instances
        """
        self.loggers = loggers
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log to all loggers"""
        for logger in self.loggers:
            logger.log(metrics, step)
    
    def close(self):
        """Close all loggers"""
        for logger in self.loggers:
            if hasattr(logger, 'close'):
                logger.close()
            elif hasattr(logger, 'finish'):
                logger.finish()
