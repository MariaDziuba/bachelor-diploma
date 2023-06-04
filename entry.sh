#!/bin/bash
nvidia-smi
python3 ./inference.py
python3 ./LIVE/LIVE/main.py --config ./modified_config.yaml --experiment experiment --signature 1 --target ./1.png
