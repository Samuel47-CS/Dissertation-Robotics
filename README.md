# Learning Stylised Bimanual Manipulation from Demonstration

This repository contains the code and experiments for my dissertation project exploring whether stylistic variation in human demonstrations can be learned and reproduced by a bimanual robotic system performing a domestic manipulation task.

## Project Overview

Robotic policies are typically trained to optimise task completion, often converging to a single "best" behaviour. However, many domestic tasks (e.g., folding, cleaning, organising) can be performed in multiple valid ways that reflect individual user preferences.

This project investigates whether:
- stylistic variation can be extracted from demonstrations, and
- encoded into robotic policies,
- without relying on explicit semantic labels.

The core idea is to treat style as variation in motion trajectories, rather than differences in task outcome.

## Setup

Clone the repository:

```bash
git clone https://github.com/Samuel47-CS/Dissertation-Robotics.git
cd Robotics
pip install -r training/requirements.txt
```

## Method

1. Collect demonstrations using teleoperation / record
2. Extract trajectories from joint positions 
3. Cluster trajectories into styles
4. Train policies on style-specific data
5. Evaluate task success and behavioural differences

## Repository Structure
.

├── clustering/           # Dataframe extraction from Hugging Face datasets, as well as data analysis and clusetering scripts

├── data/                 # Where datasets should live. Contains scripts for dataset manipulation

├── lerobot/              # Pointer to LeRobot github submodule

├── models/               # Where models should live. Contains a script to manually rename model attributes (e.g., camera names)

├── scripts/              # LeRobot CLI command bash scripts for all required functionalities required for this project

├── videos/               # Videos of trained Bi-SO101 arms with different styles.

└── README.md

### Useful Links:
- Hugging Face account that contains the full dataset and all models used in this project: https://huggingface.co/the-sam-uel
- LeRobot github: https://github.com/huggingface/lerobot 