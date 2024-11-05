#!/bin/bash


# FOR USE WITH HEX container is: tc2034/frapa2
# hare run -it --gpus device=8  -v "$(pwd)":/app tc2034/frapa2 bash

. ./start_scripts/PNN_continual_max_opt_ninja_starpilot.sh

. ./start_scripts/PNN_continual_max_opt_starpilot_climber.sh

. ./start_scripts/PNN_continual_max_opt_ninja_coinrun.sh



