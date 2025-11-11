#!/usr/bin/env python3

"""
Habitat 2 to Habitat 3 compatibility layer for StreamVLN.

This module provides compatibility shims to adapt StreamVLN code
(built on Habitat 2) to work with Habitat 3 API.

Key differences between Habitat 2 and 3:
1. Observation space changes
2. Sensor API updates
3. Task configuration structure
4. Simulator interface changes
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch


class Habitat2To3Adapter:
    """
    Adapter class to handle differences between Habitat 2 and 3.
    
    This class provides methods to:
    - Convert observation formats
    - Adapt sensor outputs
    - Handle deprecated APIs
    """
    
    @staticmethod
    def adapt_observations(obs: Dict[str, Any], obs_space: Any) -> Dict[str, Any]:
        """
        Adapt observations from Habitat 3 format to Habitat 2 format.
        
        Args:
            obs: Observations from Habitat 3
            obs_space: Observation space definition
        
        Returns:
            Adapted observations compatible with StreamVLN
        """
        adapted = {}
        
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                # Ensure correct shape and dtype
                adapted[key] = value
            elif isinstance(value, np.ndarray):
                adapted[key] = torch.from_numpy(value)
            else:
                adapted[key] = value
        
        # Ensure RGB is in correct format (H, W, C) with values in [0, 255]
        if 'rgb' in adapted:
            rgb = adapted['rgb']
            if isinstance(rgb, torch.Tensor):
                # Check if normalized
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).to(torch.uint8)
                # Ensure (H, W, C) format
                if rgb.shape[0] == 3:  # (C, H, W)
                    rgb = rgb.permute(1, 2, 0)
            adapted['rgb'] = rgb
        
        # Handle depth sensor if present
        if 'depth' in adapted:
            depth = adapted['depth']
            if isinstance(depth, torch.Tensor):
                # Ensure depth has channel dimension
                if len(depth.shape) == 2:
                    depth = depth.unsqueeze(-1)
            adapted['depth'] = depth
        
        return adapted
    
    @staticmethod
    def adapt_actions(action: Any, action_space: Any) -> Any:
        """
        Adapt actions from StreamVLN format to Habitat 3 format.
        
        Args:
            action: Action from StreamVLN
            action_space: Action space definition
        
        Returns:
            Adapted action compatible with Habitat 3
        """
        # StreamVLN actions: 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
        # Map to Habitat 3 action space as needed
        return action
    
    @staticmethod
    def get_camera_intrinsics(sim_config: Optional[Any] = None) -> np.ndarray:
        """
        Get camera intrinsic matrix.
        
        Args:
            sim_config: Simulator configuration (optional)
        
        Returns:
            4x4 camera intrinsic matrix
        """
        # Default VLN camera intrinsics
        # These values are from the original StreamVLN code
        intrinsic = np.array([
            [192.0, 0.0, 191.42857143, 0.0],
            [0.0, 192.0, 191.42857143, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        if sim_config is not None:
            # Try to extract from config if available
            try:
                if hasattr(sim_config, 'agents'):
                    agent = sim_config.agents[0]
                    if hasattr(agent, 'sim_sensors'):
                        for sensor in agent.sim_sensors.values():
                            if hasattr(sensor, 'hfov') and hasattr(sensor, 'width'):
                                # Calculate intrinsics from FOV
                                hfov = np.deg2rad(sensor.hfov)
                                width = sensor.width
                                height = sensor.height
                                focal = width / (2 * np.tan(hfov / 2))
                                intrinsic = np.array([
                                    [focal, 0.0, width / 2, 0.0],
                                    [0.0, focal, height / 2, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]
                                ])
                                break
            except Exception as e:
                warnings.warn(f"Failed to extract camera intrinsics from config: {e}")
        
        return intrinsic
    
    @staticmethod
    def adapt_config(config: Any) -> Any:
        """
        Adapt configuration from Habitat 3 to Habitat 2 format if needed.
        
        Args:
            config: Configuration object
        
        Returns:
            Adapted configuration
        """
        # For now, just return as-is
        # Add specific adaptations as needed
        return config


class DepthCameraFilter:
    """
    Depth camera filtering utilities.
    
    StreamVLN uses depth camera filtering in some scenarios.
    This class provides compatible filtering methods.
    """
    
    @staticmethod
    def filter_depth(
        depth: np.ndarray,
        min_depth: float = 0.0,
        max_depth: float = 10.0
    ) -> np.ndarray:
        """
        Filter depth values to valid range.
        
        Args:
            depth: Depth map (H, W) or (H, W, 1)
            min_depth: Minimum valid depth
            max_depth: Maximum valid depth
        
        Returns:
            Filtered depth map
        """
        filtered = np.copy(depth)
        filtered[filtered < min_depth] = min_depth
        filtered[filtered > max_depth] = max_depth
        return filtered


def ensure_streamvln_dependencies():
    """
    Ensure all StreamVLN dependencies are available.
    
    This function checks for required packages and provides
    helpful error messages if they're missing.
    """
    required_packages = {
        'transformers': 'transformers',
        'torch': 'torch',
        'PIL': 'Pillow',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        raise ImportError(
            f"Missing required packages for StreamVLN: {', '.join(missing)}\n"
            f"Please install them with: pip install {' '.join(missing)}"
        )
    
    # Check for optional but recommended packages
    optional_packages = {
        'flash_attn': 'flash-attn',
    }
    
    for module_name, package_name in optional_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            warnings.warn(
                f"Optional package {package_name} not found. "
                f"StreamVLN will work but may be slower. "
                f"Install with: pip install {package_name}"
            )


# Convenience function for quick adaptation
def adapt_for_streamvln(observations: Dict[str, Any], obs_space: Any) -> Dict[str, Any]:
    """
    Quick adaptation function for observations.
    
    Args:
        observations: Raw observations from Habitat 3
        obs_space: Observation space
    
    Returns:
        Adapted observations for StreamVLN
    """
    return Habitat2To3Adapter.adapt_observations(observations, obs_space)
