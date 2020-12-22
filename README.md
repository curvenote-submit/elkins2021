# pyUsercalc

A set of  Jupyter notebooks to replicate the functionality of Marc Spiegelman&#39;s UserCalc calculator in Python,
and to add disequilibrium transport and scaled reactive transport calculators. Determines U-series disequilibria
in basalts during decompression partial melting.

## Contents

* `pyUserCalc_manuscript.ipynb`: primary notebook describing derivation, implementation and multiple examples for running the U-series calculators
* `UserCalc.py`:  python file containing the UserCalc driver and model classes as well as some convenience visualization methods.  Imported in the notebooks
*  `data`: directory containing input .csv files

#### Additional materials

* `pyUserCalc-v3.1.ipynb`: Simplified version of the manuscript notebook intended for production work with the model.
* `latex`: directory containing latex version and formatted pdf of the manuscript notebook for publication

## Installation/Running

#### ENKI Cloud Server:
Instructions:

* Log in to the  [ENKI cloud server](https://server.enki-portal.org/hub/login) (you will need a gitlab account)
* click "Close this Screen" button at the bottom of the welcome screen
* Launch a terminal from the Launcher
* Clone the repository to your server
`git clone https://gitlab.com/ENKI-portal/pyUsercalc.git`
* navigate to the `pyUserCalc` directory and open `pyUserCalc_manuscript.ipynb`



#### Binder


You can also interact with the notebook through a binder container. Click the button below

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/ENKI-portal%2FpyUsercalc/master?filepath=pyUserCalc_manuscript.ipynb)

and a container will be built to run the manuscript notebook through your browser.  This may take some time to start however.


#### Local Install

This repository should run with a standard Jupyter installation e.g. through Anaconda (python3).  Just clone the repository and open in jupyter-lab. 



