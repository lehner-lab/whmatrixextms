#!/bin/bash

FITNESS_FILE="Data/tRNA/JD_Phylogeny_tR-R-CCU_dimsum1.3_fitness_replicates.txt"
OUTPUT_DIR="output_tRNA"
DATASET="trna"
PLOT_DIR="plots_tRNA"
VALIDATION_FILE="Data/Validations/EpGlobal_FDRall_reformat.txt"

./whmatrixextms_plot.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --plot_dir $PLOT_DIR --dataset $DATASET --validation_file $VALIDATION_FILE

