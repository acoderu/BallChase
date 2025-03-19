#!/usr/bin/env python3

"""
Configuration Loader Utility
----------------------------
This module provides utilities for loading and managing configuration parameters
for the tennis ball tracking system, ensuring consistent parameter handling
across all nodes.
"""

import os
import yaml
import rclpy.node
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Loads configuration parameters from YAML files and provides
    them to ROS nodes with proper parameter declaration.
    """
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            file_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the configuration parameters
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"ERROR: Configuration file not found: {file_path}")
            raise
        except yaml.YAMLError as e:
            print(f"ERROR: Invalid YAML in configuration file: {file_path}")
            print(f"Details: {str(e)}")
            raise

    @staticmethod
    def declare_parameters_from_config(node: rclpy.node.Node, config: Dict[str, Any], namespace: str = "") -> None:
        """
        Declare ROS parameters from a configuration dictionary.
        
        Args:
            node: The ROS node to declare parameters on
            config: Dictionary containing configuration parameters
            namespace: Optional namespace for parameters
        """
        for key, value in config.items():
            param_name = f"{namespace}.{key}" if namespace else key
            
            # Handle nested dictionaries recursively
            if isinstance(value, dict):
                ConfigLoader.declare_parameters_from_config(node, value, param_name)
            else:
                node.declare_parameter(param_name, value)
                
    @staticmethod
    def load_and_declare_parameters(node: rclpy.node.Node, config_name: str, 
                                    config_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a configuration file and declare its parameters to a ROS node.
        
        Args:
            node: The ROS node to declare parameters on
            config_name: Name of the configuration file (without .yaml extension)
            config_dir: Directory containing config files (default: environment variable or current dir)
            
        Returns:
            Dictionary containing the loaded configuration
        """
        # Determine config directory
        if config_dir is None:
            config_dir = os.environ.get('BALL_CHASE_CONFIG_DIR', 
                             os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config'))
        
        # Load the config file
        config_path = os.path.join(config_dir, f"{config_name}.yaml")
        config = ConfigLoader.load_yaml(config_path)
        
        # Declare parameters
        ConfigLoader.declare_parameters_from_config(node, config)
        
        return config
