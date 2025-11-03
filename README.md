# Solvent Exposure

Simple tools to handle pdbs and calculate solvent exposure scores

## Installation
### Running in python

I recommend running this in its own environment. I will give examples for Anaconda, but use your own systems and pip if preferred.

```python
conda create -n solvent_exposure pyqt=5
```

PyQt is not strictly necessary for only running in command line, as .py files, or in a notebook (.ipynb files). Feel free to omit if you want

Activate the environment and install the other dependencies:
```python
conda activate solvent_exposure
```

```python
conda install -c conda-forge biopandas
```

```python
conda install matplotlib
```

If you wish to the .ipynb file in VS code or similar, install ipykernel
```python
conda install ipykernel
```

If you wish to compile your code, with your own modifications and default settings, into a standalone app, I recommend installing pyinstaller
```python
conda install -c conda-forge pyinstaller
```

### Running as a standalone app

It is, in principle, possible to run the GUI version as a standalone app without requiring local python installation. This is designed to make it easier for more users to calculate solvent exposures, but has a few downsides, for example:
* It is not possible to change default settings
* There are currently a limited number of calculations that can be run in GUI mode. I hope to make it more flexible as time allows, but running custom calculations for your own purposes is done more easily in python
* Externally generated .exe files are sometimes flagged as a potential virus 