"""
Seed manager for ensuring reproducibility across all components.
"""
import random
import numpy as np
import torch
import os
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Set random seed for all random number generators.
    
    Args:
        seed: Random seed value
        deterministic: If True, ensures deterministic behavior (may be slower)
        benchmark: If True, enables cuDNN benchmarking (faster but non-deterministic)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set number of threads for reproducibility
        torch.set_num_threads(1)
        
        logger.info(f"Seed set to {seed} with deterministic mode enabled")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
        
        logger.info(f"Seed set to {seed} with non-deterministic mode")


class ReproducibilityManager:
    """Manage reproducibility settings across experiments."""
    
    def __init__(self, seed: int = 42, deterministic: bool = True):
        """
        Initialize reproducibility manager.
        
        Args:
            seed: Random seed
            deterministic: Enable deterministic mode
        """
        self.seed = seed
        self.deterministic = deterministic
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize reproducibility settings."""
        if self._initialized:
            logger.warning("Reproducibility already initialized")
            return
        
        set_seed(self.seed, self.deterministic)
        self._initialized = True
        
        logger.info("Reproducibility manager initialized")
    
    def get_generator(self, device: str = "cpu") -> torch.Generator:
        """
        Get a seeded PyTorch generator.
        
        Args:
            device: Device for the generator ('cpu' or 'cuda')
            
        Returns:
            Seeded torch.Generator
        """
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        return generator
    
    def get_numpy_rng(self) -> np.random.Generator:
        """
        Get a seeded NumPy random number generator.
        
        Returns:
            Seeded np.random.Generator
        """
        return np.random.default_rng(self.seed)
    
    def worker_init_fn(self, worker_id: int) -> None:
        """
        Worker initialization function for DataLoader.
        Ensures each worker has a different but reproducible seed.
        
        Args:
            worker_id: Worker process ID
        """
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def verify_reproducibility(model, input_tensor, n_runs: int = 3) -> bool:
    """
    Verify that model produces identical outputs across multiple runs.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor for testing
        n_runs: Number of runs to verify
        
    Returns:
        True if outputs are identical, False otherwise
    """
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            output = model(input_tensor)
            outputs.append(output.cpu())
    
    # Check if all outputs are identical
    for i in range(1, n_runs):
        if not torch.allclose(outputs[0], outputs[i], rtol=1e-5, atol=1e-8):
            logger.error(f"Reproducibility check failed: run 0 and run {i} differ")
            return False
    
    logger.info(f"Reproducibility verified across {n_runs} runs")
    return True


if __name__ == "__main__":
    # Test reproducibility
    print("Testing reproducibility...")
    
    # Test 1: NumPy
    set_seed(42)
    arr1 = np.random.rand(10)
    
    set_seed(42)
    arr2 = np.random.rand(10)
    
    assert np.allclose(arr1, arr2), "NumPy reproducibility failed"
    print("✓ NumPy reproducibility verified")
    
    # Test 2: PyTorch
    set_seed(42)
    tensor1 = torch.randn(10)
    
    set_seed(42)
    tensor2 = torch.randn(10)
    
    assert torch.allclose(tensor1, tensor2), "PyTorch reproducibility failed"
    print("✓ PyTorch reproducibility verified")
    
    # Test 3: Python random
    set_seed(42)
    rand1 = [random.random() for _ in range(10)]
    
    set_seed(42)
    rand2 = [random.random() for _ in range(10)]
    
    assert rand1 == rand2, "Python random reproducibility failed"
    print("✓ Python random reproducibility verified")
    
    print("\nAll reproducibility tests passed!")
