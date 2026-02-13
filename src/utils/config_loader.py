"""
Configuration loader utility.
Handles loading and validation of YAML configuration files.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Root directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise ValueError(f"Configuration directory not found: {config_dir}")
    
    def load_config(self, module: str, config_name: str = "config.yaml") -> Dict[str, Any]:
        """
        Load configuration for a specific module.
        
        Args:
            module: Module name (vision, market, fusion, data)
            config_name: Name of the config file
            
        Returns:
            Dictionary containing configuration
        """
        config_path = self.config_dir / module / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        self._validate_config(config, module)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any], module: str) -> None:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary
            module: Module name for context-specific validation
        """
        if not isinstance(config, dict):
            raise ValueError(f"Invalid configuration format for {module}")
        
        # Module-specific validation
        if module == "vision":
            required_keys = ["model", "training", "data"]
        elif module == "market":
            required_keys = ["model", "features", "training"]
        elif module == "fusion":
            required_keys = ["inputs", "fusion", "valuation_head"]
        elif module == "data":
            required_keys = ["sources", "filters", "storage"]
        else:
            required_keys = []
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        logger.debug(f"Configuration for {module} validated successfully")
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        Later configs override earlier ones.
        
        Args:
            *configs: Variable number of configuration dictionaries
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        for config in configs:
            merged = self._deep_merge(merged, config)
        return merged
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Recursively merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary with updates
            
        Returns:
            Merged dictionary
        """
        merged = base.copy()
        
        for key, value in update.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_path}")


def load_config(module: str, config_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        module: Module name (vision, market, fusion, data)
        config_name: Optional specific config file name
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader()
    
    # Determine config file name
    if config_name is None:
        config_map = {
            "vision": "cnn_config.yaml",
            "market": "temporal_config.yaml",
            "fusion": "fusion_config.yaml",
            "data": "data_config.yaml"
        }
        config_name = config_map.get(module, "config.yaml")
    
    return loader.load_config(module, config_name)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_loader.py <module>")
        sys.exit(1)
    
    module = sys.argv[1]
    config = load_config(module)
    print(f"\nConfiguration for {module}:")
    print(yaml.dump(config, default_flow_style=False))
