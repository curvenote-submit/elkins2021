# pyUsercalc

A set of Jupyter notebooks to replicate the functionality of Marc Spiegelman&#39;s UserCalc calculator in Python,
and to add disequilibrium transport and scaled reactive transport calculators. Determines U-series disequilibria
in basalts during decompression partial melting.

## Contents

* `pyUserCalc_manuscript.ipynb`: Primary Elkins and Spiegelman (submitted) manuscript notebook describing derivations, model implementation, and multiple examples for running the U-series calculators.
* `UserCalc.py`:  Python file containing the UserCalc driver and model classes as well as some convenient visualization methods, and which is imported in the model notebooks.
*  `data`: Directory containing input .csv files. These files are necessary for the primary notebook to run, and can also be used as templates for additional model use.

#### Additional materials

* `pyUserCalc-v3.1.ipynb`: Simplified version of the primary Elkins and Spiegelman (submitted) pyUserCalc notebook, intended for production work with the model.
* `latex`: Directory containing LaTeX version and formatted PDF of the Elkins and Spiegelman (submitted) manuscript for publication.

## Installation/Running

#### ENKI Cloud Server:
Instructions:

* Register for a free [GitLab account](https://gitlab.com/ENKI-portal), if you do not already have one.
* Use your GitLab account to log in to the [ENKI cloud server](https://server.enki-portal.org/hub/login). Note that logging in for the first time can take a few minutes while your server is being built.
* Once you are logged in, click "Close this Screen" button at the bottom of the welcome screen.
* In the Launcher tab, use the "Terminal" button to launch a terminal window.
* At the terminal prompt, type the following text to clone the pyUserCalc code repository to your own server, and then type Return:
`git clone https://gitlab.com/ENKI-portal/pyUsercalc.git`
* Now navigate to the `pyUserCalc` directory in the left sidebar panel and double-click to open it. To access the primary manuscript notebook, open the file `pyUserCalc_manuscript.ipynb`, again by double clicking. To access the simplified notebook version, instead open the file `pyUserCalc-v3.1.ipynb`.



#### Binder


You can also interact directly with the manuscript notebook through a binder container. To do this, click the button below:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/ENKI-portal%2FpyUsercalc/master?filepath=pyUserCalc_manuscript.ipynb)

and a container will be built that can run the manuscript notebook through your own browser. This may take a few moments to start.


#### Local Install

This repository should run with a standard Jupyter installation e.g. through Anaconda (python3).  To do so, clone the repository and open it in JupyterLab. 



