#!/bin/bash

FITNESS_FILE="Data/eqFP611/41467_2019_12130_MOESM7_ESM_reformat.txt"
OUTPUT_DIR="output_eqFP611"
DATASET="eqFP611"
PLOT_DIR="plots_eqFP611"
VALIDATION_FILE="Data/Validations/41467_2019_12130_MOESM7_ESM_reformat_coef.txt"

./whmatrixextms_plot.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --plot_dir $PLOT_DIR --dataset $DATASET --validation_file $VALIDATION_FILE

