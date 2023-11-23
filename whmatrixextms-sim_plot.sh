#!/bin/bash

FITNESS_FILE="Data/Validations/simulated_comb6mer_BA_noise1.txt"
OUTPUT_DIR="output_simulated_BA_noise1"
DATASET="simulated"
PLOT_DIR="plots_simulated_BA_noise1"
VALIDATION_FILE="Data/Validations/simulated_comb6mer_BA_noise1.txt"

./whmatrixextms_plot.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --plot_dir $PLOT_DIR --dataset $DATASET --validation_file $VALIDATION_FILE

FITNESS_FILE="Data/Validations/simulated_comb6mer_BA_noise2.txt"
OUTPUT_DIR="output_simulated_BA_noise2"
DATASET="simulated"
PLOT_DIR="plots_simulated_BA_noise2"
VALIDATION_FILE="Data/Validations/simulated_comb6mer_BA_noise2.txt"

./whmatrixextms_plot.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --plot_dir $PLOT_DIR --dataset $DATASET --validation_file $VALIDATION_FILE

FITNESS_FILE="Data/Validations/simulated_comb6mer_BA_noise3.txt"
OUTPUT_DIR="output_simulated_BA_noise3"
DATASET="simulated"
PLOT_DIR="plots_simulated_BA_noise3"
VALIDATION_FILE="Data/Validations/simulated_comb6mer_BA_noise3.txt"

./whmatrixextms_plot.py --fitness_file $FITNESS_FILE --output_dir $OUTPUT_DIR --plot_dir $PLOT_DIR --dataset $DATASET --validation_file $VALIDATION_FILE
