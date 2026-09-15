#!/bin/bash

# Linux example: run multiple training commands with different dim/epoch settings.
python main.py --dim 160 --epoch 100 --batch_size 128 --lr 1e-5
python main.py --dim 128 --epoch 100 --batch_size 128 --lr 5e-5
python main.py --dim 128 --epoch 100 --batch_size 128 --lr 5e-4
python main.py --dim 128 --epoch 100 --batch_size 128 --lr 1e-3