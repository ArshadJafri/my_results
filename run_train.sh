#!/bin/bash

deepspeed --num_gpus=8 Finetuning_V5.py 0.00007 9216 2 2 2 0.05 15




