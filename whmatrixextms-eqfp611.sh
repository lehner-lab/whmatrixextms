#!/bin/bash

FITNESS_FILE="Data/eqFP611/41467_2019_12130_MOESM7_ESM_reformat.txt"
OUTPUT_DIR="output_eqFP611"
DATASET="eqFP611"
MAX_INT_ORDER="6"

#$ -t 1-100
./whmatrixextms.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --seed ${SGE_TASK_ID} --dataset $DATASET --max_interaction_order $MAX_INT_ORDER

