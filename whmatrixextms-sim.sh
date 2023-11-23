#!/bin/bash

FITNESS_FILE="Data/Validations/simulated_comb6mer_BA_noise1.txt"
OUTPUT_DIR="output_simulated_BA_noise1"
DATASET="simulated"
MAX_INT_ORDER="3"

#$ -t 1-100
./whmatrixextms.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --seed ${SGE_TASK_ID} --dataset $DATASET --max_interaction_order $MAX_INT_ORDER

FITNESS_FILE="Data/Validations/simulated_comb6mer_BA_noise2.txt"
OUTPUT_DIR="output_simulated_BA_noise2"
DATASET="simulated"
MAX_INT_ORDER="3"

#$ -t 1-100
./whmatrixextms.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --seed ${SGE_TASK_ID} --dataset $DATASET --max_interaction_order $MAX_INT_ORDER

FITNESS_FILE="Data/Validations/simulated_comb6mer_BA_noise3.txt"
OUTPUT_DIR="output_simulated_BA_noise3"
DATASET="simulated"
MAX_INT_ORDER="3"

#$ -t 1-100
./whmatrixextms.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --seed ${SGE_TASK_ID} --dataset $DATASET --max_interaction_order $MAX_INT_ORDER

