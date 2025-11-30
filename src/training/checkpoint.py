"""
Checkpoint management for saving/loading model state
"""

import os
from pathlib import Path
from flax.training import checkpoints
from flax import serialization
import jax
from typing import Any, Optional
import json


class CheckpointManager:
    """
    Manages model checkpointing
    
    Features:
    - Automatic checkpoint saving
    - Keep only N latest checkpoints
    - Async saving (non-blocking)
    - Metadata tracking
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 3,
        save_interval_steps: int = 1000,
        async_save: bool = True
    ):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
            save_interval_steps: Save every N steps
            async_save: Use async saving
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_to_keep = max_to_keep
        self.save_interval_steps = save_interval_steps
        self.async_save = async_save
        
        print(f"Checkpoint manager initialized:")
        print(f"  Directory: {self.checkpoint_dir}")
        print(f"  Max to keep: {max_to_keep}")
        print(f"  Save interval: {save_interval_steps} steps")
    
    def save(
        self,
        state: Any,
        step: int,
        metadata: Optional[dict] = None,
        force: bool = False
    ):
        """
        Save checkpoint
        
        Args:
            state: Training state to save
            step: Current step
            metadata: Optional metadata dict
            force: Force save even if not at interval
        """
        # Check if should save
        if not force and step % self.save_interval_steps != 0:
            return
        
        # Use unreplicated state if distributed
        if hasattr(state, 'step') and isinstance(state.step, jax.Array):
            try:
                # Try to unreplicate if needed
                from .distributed import unreplicate_from_devices
                state = unreplicate_from_devices(state)
            except:
                pass
        
        # Save checkpoint
        checkpoints.save_checkpoint(
            ckpt_dir=str(self.checkpoint_dir),
            target=state,
            step=step,
            keep=self.max_to_keep,
            overwrite=False,
            async_manager=None  # Can add async manager here
        )
        
        # Save metadata
        if metadata:
            metadata_path = self.checkpoint_dir / f"checkpoint_{step}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✓ Checkpoint saved at step {step}")
    
    def restore(
        self,
        state: Any,
        step: Optional[int] = None
    ) -> Any:
        """
        Restore checkpoint
        
        Args:
            state: Template state (for structure)
            step: Specific step to restore (None = latest)
        
        Returns:
            Restored state
        """
        restored = checkpoints.restore_checkpoint(
            ckpt_dir=str(self.checkpoint_dir),
            target=state,
            step=step
        )
        
        if restored is state:
            print(f"⚠ No checkpoint found in {self.checkpoint_dir}")
            return state
        
        actual_step = restored.step if hasattr(restored, 'step') else step
        print(f"✓ Checkpoint restored from step {actual_step}")
        
        return restored
    
    def latest_checkpoint_step(self) -> Optional[int]:
        """Get latest checkpoint step number"""
        latest = checkpoints.latest_checkpoint(str(self.checkpoint_dir))
        if latest:
            # Extract step from path
            try:
                return int(latest.split('_')[-1])
            except:
                return None
        return None
    
    def list_checkpoints(self) -> list:
        """List all checkpoint steps"""
        checkpoints_list = []
        for path in self.checkpoint_dir.glob("checkpoint_*"):
            if path.is_dir() or path.suffix == '':
                try:
                    step = int(path.name.split('_')[-1])
                    checkpoints_list.append(step)
                except:
                    pass
        return sorted(checkpoints_list)
    
    def delete_checkpoint(self, step: int):
        """Delete specific checkpoint"""
        ckpt_path = self.checkpoint_dir / f"checkpoint_{step}"
        if ckpt_path.exists():
            import shutil
            shutil.rmtree(ckpt_path)
            print(f"✓ Deleted checkpoint at step {step}")
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_to_keep"""
        steps = self.list_checkpoints()
        if len(steps) > self.max_to_keep:
            for step in steps[:-self.max_to_keep]:
                self.delete_checkpoint(step)


def save_checkpoint_simple(
    checkpoint_dir: str,
    state: Any,
    step: int,
    max_to_keep: int = 3
):
    """
    Simple checkpoint save function
    
    Args:
        checkpoint_dir: Directory path
        state: State to save
        step: Current step
        max_to_keep: Max checkpoints to keep
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=step,
        keep=max_to_keep
    )


def restore_checkpoint_simple(
    checkpoint_dir: str,
    state: Any,
    step: Optional[int] = None
) -> Any:
    """
    Simple checkpoint restore function
    
    Args:
        checkpoint_dir: Directory path
        state: Template state
        step: Specific step (None = latest)
    
    Returns:
        Restored state
    """
    if not os.path.exists(checkpoint_dir):
        return state
    
    return checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=step
    )
