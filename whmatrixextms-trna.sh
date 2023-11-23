#!/bin/bash

FITNESS_FILE="Data/tRNA/JD_Phylogeny_tR-R-CCU_dimsum1.3_fitness_replicates.txt"
OUTPUT_DIR="output_tRNA"
DATASET="trna"
MAX_INT_ORDER="8"

#$ -t 1-100
./whmatrixextms.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --seed ${SGE_TASK_ID} --dataset $DATASET --max_interaction_order $MAX_INT_ORDER

