# HRC 2026 Official Resources

This file is the canonical list of official external resources used by the HRC 2026 project.  
It exists to help future contributors and agents distinguish clearly between the baseline codebase, simulation assets, dataset resources, and the Walker S2 robot model source.

## Baseline

**Name:** Baseline  
**Link:** https://github.com/UBTECH-Robot/GlobalHumanoidRobotChallenge_2026_Baseline  
**Description:** A baseline algorithm package that includes the standard workflow and reference paradigm for model training and deployment.  
**Role in this project:** Reference codebase for training, deployment, and overall workflow inspection. This is the main starting point for understanding the official baseline pipeline.

## Assets

**Name:** Assets  
**Link:** https://huggingface.co/UBTECH-Robotics/challenge2026_assets  
**Description:** Simulation assets.  
**Role in this project:** Official simulation assets for scenes, task objects, and environment resources used by the competition stack.

## Dataset

**Name:** Dataset  
**Link:** https://huggingface.co/datasets/UBTECH-Robotics/challenge2026_dataset  
**Description:** Official challenge dataset resource, currently used here as the available Packing_Box dataset reference.  
**Role in this project:** Official dataset resource currently available for inspection and possible training/bootstrap use. It should not be confused with the environment assets or baseline code.

## Walkers2usd

**Name:** Walkers2usd  
**Link:** https://github.com/UBTECH-Robot/WalkerS2-Model-Challenge  
**Description:** The USD model of the Walker S2 robot, adapted for the Isaac Sim simulation platform.  
**Role in this project:** Source of the Walker S2 robot USD/URDF/STL assets and related robot model resources for Isaac Sim integration.
