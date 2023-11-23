Welcome to the GitHub repository for the following publication: [An extension of the Walsh-Hadamard transform to calculate and model epistasis in genetic landscapes of arbitrary shape and complexity (Faure AJ, et al., 2023)](https://www.biorxiv.org/content/10.1101/2023.03.06.531391)

Here you'll find the Python code to reproduce the figures and results from the computational analyses described in the paper.

# Contents

* **1. [whmatrixextms-benchmarkin.ipynb](whmatrixextms-benchmarking.ipynb)** Jupyter Notebook to perform benchmarking analyses and plot heat map representations of matrices.
* **2. [whmatrixextms-validations.ipynb](whmatrixextms-validations.ipynb)** Jupyter Notebook to reformat published background-averaged epistatic coefficients and simulate multiallelic genetic landscape.
* **3. [whmatrixextms.py](whmatrixextms.py)** Python script to fit sparse models to the fitness landscapes.
* **4. [whmatrixextms_plot.py](whmatrixextms_plot.py)** Python script to plot model results.

# Required Software

You will need the following dependencies installed:

* **[_Python_](https://www.python.org/) >=v3.9.9** (NumPy, pandas, scikit-learn, SciPy, Matplotlib, seaborn>=v0.12)

# Required Data

DMS data (fitness estimates) and additional files required to run the above analyses can be downloaded from [here](https://www.dropbox.com/scl/fi/441zb6avxczuer6z6axb6/Data.zip?rlkey=73we7pwojjc4n756nostlb4wa&dl=0).

Bash scripts with command-line options for fitting sparse models to fitness landscape for each dataset are also included in this repository.

