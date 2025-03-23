import os
import yaml
from ament_index_python.packages import get_package_share_directory

class ConfigLoader:
    def __init__(self):
        self.package_name = 'ball_chase'
        self.config_dir = os.path.join(get_package_share_directory(self.package_name), 'config')
    
    def load_yaml(self, filename):
        """Load a YAML configuration file from the config directory"""
        filepath = os.path.join(self.config_dir, filename)
        try:
            with open(filepath, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config file {filepath}: {e}")
            return {}