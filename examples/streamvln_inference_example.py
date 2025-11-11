#!/usr/bin/env python3

"""
StreamVLN Inference Example

This script demonstrates how to use the StreamVLN policy
integrated into the Falcon framework for visual-language navigation.

Usage:
    python examples/streamvln_inference_example.py \
        --model-path /path/to/streamvln/model \
        --instruction "go to the kitchen"
"""

import argparse
import numpy as np
import torch
from PIL import Image
from gym import spaces

from habitat_baselines.rl.ddppo.policy.streamvln_policy import StreamVLNNet


def create_dummy_observation_space():
    """Create a dummy observation space for demonstration."""
    return spaces.Dict({
        'rgb': spaces.Box(
            low=0, high=255,
            shape=(480, 640, 3),
            dtype=np.uint8
        ),
        'depth': spaces.Box(
            low=0.0, high=10.0,
            shape=(480, 640, 1),
            dtype=np.float32
        ),
    })


def create_dummy_action_space():
    """Create a dummy action space for demonstration."""
    # StreamVLN uses 4 discrete actions: STOP, FORWARD, LEFT, RIGHT
    return spaces.Discrete(4)


def main():
    parser = argparse.ArgumentParser(description='StreamVLN Inference Example')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to pretrained StreamVLN model'
    )
    parser.add_argument(
        '--instruction',
        type=str,
        default='move forward and turn left',
        help='Navigation instruction'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='Number of steps to simulate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    parser.add_argument(
        '--image-path',
        type=str,
        default=None,
        help='Optional: Path to RGB image to use (default: random noise)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("StreamVLN Inference Example")
    print("=" * 80)
    print(f"Model Path: {args.model_path}")
    print(f"Instruction: {args.instruction}")
    print(f"Device: {args.device}")
    print(f"Number of steps: {args.num_steps}")
    print("=" * 80)
    
    # Create observation and action spaces
    obs_space = create_dummy_observation_space()
    action_space = create_dummy_action_space()
    
    # Initialize StreamVLN network
    print("\nInitializing StreamVLN model...")
    model = StreamVLNNet(
        observation_space=obs_space,
        action_space=action_space,
        hidden_size=1024,
        model_path=args.model_path,
        num_frames=32,
        num_history=8,
        num_future_steps=4,
        model_max_length=4096,
        device=args.device,
        discrete_actions=True,
    )
    
    print("Model initialized successfully!")
    
    # Load or create RGB image
    if args.image_path:
        print(f"\nLoading image from {args.image_path}")
        rgb = np.array(Image.open(args.image_path).resize((640, 480)))
    else:
        print("\nUsing random noise image (for demonstration)")
        # Create a simple gradient image for visualization
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(480):
            for j in range(640):
                rgb[i, j, 0] = (i * 255) // 480  # Red gradient
                rgb[i, j, 1] = (j * 255) // 640  # Green gradient
                rgb[i, j, 2] = 128  # Constant blue
    
    print(f"RGB image shape: {rgb.shape}")
    
    # Run inference for multiple steps
    print("\n" + "=" * 80)
    print("Running Inference")
    print("=" * 80)
    
    action_names = {
        0: 'STOP',
        1: 'MOVE_FORWARD (↑)',
        2: 'TURN_LEFT (←)',
        3: 'TURN_RIGHT (→)'
    }
    
    all_actions = []
    
    for step in range(args.num_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # Determine if we should run the model or reuse cached results
        # StreamVLN generates multiple actions at once, so we only need to
        # run the model every few steps
        run_model = (step % 4 == 0)  # Run model every 4 steps
        
        try:
            action_seq, llm_output = model.generate_action_sequence(
                rgb=rgb,
                instruction=args.instruction,
                env_idx=0,
                run_model=run_model
            )
            
            if run_model:
                print(f"Generated action sequence: {action_seq}")
                print(f"Action names: {[action_names.get(a, 'UNKNOWN') for a in action_seq]}")
                print(f"\nLLM Output (first 200 chars):")
                print(llm_output[:200] + "..." if len(llm_output) > 200 else llm_output)
                
                all_actions.extend(action_seq)
            
            # Get next action
            if len(action_seq) > 0:
                action = action_seq[0]
                print(f"\nExecuting action: {action} ({action_names.get(action, 'UNKNOWN')})")
            else:
                print("\nNo action generated, defaulting to STOP")
                action = 0
        
        except Exception as e:
            print(f"\nError during inference: {e}")
            print("Defaulting to STOP action")
            action = 0
        
        # In a real environment, you would:
        # obs, reward, done, info = env.step(action)
        # rgb = obs['rgb']
        
    print("\n" + "=" * 80)
    print("Inference Complete")
    print("=" * 80)
    print(f"Total actions generated: {len(all_actions)}")
    if all_actions:
        print(f"Action sequence: {all_actions}")
        print(f"Action names: {[action_names.get(a, 'UNKNOWN') for a in all_actions]}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
