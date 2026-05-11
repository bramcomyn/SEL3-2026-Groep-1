# Brittle Star-Inspired Damage Robustness with Multi-Agent Reinforcement Learning

This repository contains our research code and documentation for the Software Engineering Lab 3 course at Ghent University.

**Group 1:** Bram Comyn, Brent Janssens & Nyah Van Wayenberge

## Overview

The goal of Software Engineering Lab 3 is to explore the world of scientific research and gain hands-on experience with the research process.
The project is mainly focused on the implementation of reinforcement learning and evolutionary algorithms, and the application of these techniques to a specific problem domain.
In this case, we are inspired by the brittle star, a marine animal known for its ability to survive and adapt to damage.

We aim to develop a multi-agent reinforcement learning framework that can learn to control a brittle star-inspired robot morphology, which can adapt to damage and continue to function effectively.
This project involves both the design, implementation and evaluation of this framework using a simulated environment, as well as the analysis of the results to understand the effectiveness of our approach.

## Documentation

The documentation for this project is organized as follows:

- **README.md** (this file): Provides an overview of the project, including its goals, structure and how to use the code.
- **docs/**: Contains detailed documentation on the design and implementation of the project, as well as the (Typst) source code of our research poster.
    API documentation is build and deployed to Github Pages using Sphinx and can be found in the `docs/api/` directory.
- **Github Wiki**: Contains additional information and resources related to the project, such as background information on the brittle star and central pattern generators, as well as multi-agent reinforcement learning.

## Installation

This project is implemented in Python and requires several dependencies to run.
We recommend using `uv` to manage the virtual environment and dependencies.
To set up the project, clone the repository and install the dependencies using the following commands:

```bash
git clone <repository-url> <project-directory>
cd <project-directory>
uv sync
```

If you have a CUDA-compatible GPU and want to use it for training, you can sync the environment using:

```bash
uv sync --extra gpu
```

## Usage

Our Python package is structured to allow for easy training and evaluation of the model.
We've made a unified command-line interface (CLI) to streamline the process of configuring train/eval modes, which configurations to use and where to save the results.
Run the following command to see the available options and usage instructions:

```shell-session
uv run -m brittle_star_locomotion --help
... # warnings from libraries that can safely be ignored
usage: __main__.py [-h] [-c CONFIG] [-v] [-m {train,eval}] [-p CHECKPOINT] [--output-video OUTPUT_VIDEO] [--render] [--output-actions-trajectory OUTPUT_ACTIONS_TRAJECTORY]
                   [--output-positions-trajectory OUTPUT_POSITIONS_TRAJECTORY]

Brittle Star Locomotion

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        path to the configuration file
  -v, --verbose         enable verbose logging
  -m {train,eval}, --mode {train,eval}
                        mode to run the project in (training or evaluation)
  -p CHECKPOINT, --checkpoint CHECKPOINT
                        path to the model checkpoint for evaluation (prefix for the checkpoint files)
  --output-video OUTPUT_VIDEO
                        path to save evaluation video
  --render              render the evaluation trajectory
  --output-actions-trajectory OUTPUT_ACTIONS_TRAJECTORY
                        path to save action trajectory csv
  --output-positions-trajectory OUTPUT_POSITIONS_TRAJECTORY
                        path to save positions trajectory csv
```

To train the model, you can use the following command:

## Training

```bash
uv run -m brittle_star_locomotion 
    --mode train
    --config <path-to-training-config> 
    --checkpoint <path-to-save-checkpoint>
```

## Evaluation

To evaluate the model and save the results in a video file, you can use the following command:

```bash
uv run -m brittle_star_locomotion 
    --mode eval
    --config <path-to-evaluation-config> 
    --checkpoint <path-to-checkpoint-for-evaluation> 
    --output-video <path-to-save-video>
    --render
```
