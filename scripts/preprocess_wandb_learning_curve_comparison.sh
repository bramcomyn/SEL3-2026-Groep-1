#!/bin/bash

RAW_WANDB_FILE=$1

cat ${RAW_WANDB_FILE} \
    | cut -d',' -f1,2,5 \
    | tr -d '\"' \
    | sed -E "s/([^0-9].*),([^0-9].*),([^0-9].*)/step,brittle_star_1,brittle_star_2/g"
