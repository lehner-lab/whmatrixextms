Welcome to the GitHub repository for the following publication: [An extension of the Walsh-Hadamard transform to calculate and model epistasis in genetic landscapes of arbitrary shape and complexity (Faure AJ, et al., 2023)](https://www.biorxiv.org/content/10.1101/2023.03.06.531391)

Here you'll find the Python code to reproduce the figures and results from the computational analyses described in the paper.

# Contents

* **1. [whmatrixextms-notebook.ipynb](whmatrixextms-notebook.ipynb)** Jupyter Notebook to perform benchmarking analyses and plot heat map representations of matrices.
* **2. [whmatrixextms-trna.py](whmatrixextms-trna.py)** Python script to fit sparse models to the fitness landscape of a tRNA.
* **3. [whmatrixextms-trna_plot.py](whmatrixextms-trna.py)** Python script to plot tRNA model results.

# Required Software

You will need the following dependencies installed:

* **[_Python_](https://www.python.org/) >=v3.9.9** (NumPy, pandas, scikit-learn, SciPy, Matplotlib, seaborn>=v0.12)

# Required Data

tRNA DMS data (fitness estimates) pre-processed with [DiMSum](https://github.com/lehner-lab/DiMSum) can be downloaded from [here](https://www.dropbox.com/s/1e6bhgi7gf6nrur/JD_Phylogeny_tR-R-CCU_dimsum1.3_fitness_replicates.RData?dl=0).

