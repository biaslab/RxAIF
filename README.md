This package defines nodes, updates and demos specific to reactive implementations of Active Inference.

# Install Julia
In order to install the Julia language (`v1.10.4`), follow the platform-specific instructions at https://julialang.org/downloads/

# Install Jupyter Notebook
Jupyter notebook is a framework for running Julia scripts (among other languages). It is well-suited for showing demo applications and interactive experimentation. In order to install Jupyter Notebook, follow the instructions at https://jupyter.readthedocs.io/en/latest/install.html

# Install required packages
The simulation notebooks require several external packages. To install them, open Julia
```
$ julia
```
and enter the package prompt by typing a closing bracket
```
julia> ]
```
Next, activate the virtual environment
```
(v1.10) pkg> activate .
```
and instantiate the required packages
```
(RxAIF) pkg> instantiate
```
This will download and install the required packages in the virtual environment named LAIF.

# Run the demos
Exit Julia, navigate to the root directory and start a Jupyter server
```
~/RxAIF$ jupyter notebook
```
A browser window should open, and you can select the demo you wish to run.

# References
- Koudahl, van de Laar & De Vries (2023). Realising Synthetic Active Inference Agents, Part I: Epistemic Objectives and Graphical Specification Language. arXiv preprint arXiv:2306.08014.
- van de Laar, Koudahl, & De Vries (2024). Realizing Synthetic Active Inference Agents, Part II: Variational Message Updates. Neural Computation.

# License
MIT License, Copyright (c) 2024 BIASlab http://biaslab.org
