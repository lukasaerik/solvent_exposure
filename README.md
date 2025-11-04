# Solvent Exposure

Simple tools to handle pdbs and calculate solvent exposure scores

## Installation
### In python

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
conda install -c conda-forge biopandas mplcursors
```

```python
conda install matplotlib
```

If you wish to run the .ipynb file in VS code or similar, install ipykernel
```python
conda install ipykernel
```

If you wish to run the .ipynb file from command line, install jupyter
```python
conda install jupyter
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

## Running
```funcs.py``` contains all standard functions, and is called by ```app.py``` and ```solvent_exposure.ipynb``` for running in a GUI or notebook, respectively.

```app.py``` generates a GUI. This is the easiest way to use these functions as intended for beginner users. There are only limited options for custom calculations, but the GUI should cover all standard use cases. Note: the interactive plot can be very slow on Mac devices with M-series processors.

```solvent_exposure.ipynb``` is a notebook version that shows how to run standard calculations as intended, and a couple advanced options as examples.

As standard, raw (input) pdb files can be placed into ```/pdbs/in/``` and calculated defattr files can be placed into ```/pdbs/out/defattr/```. Preprocessed files are automatically saved to ```/pdbs/preprocessed/``` and files where solvent exposure has been calculated are saved to ```/pdbs/out/```

## Visualising solvent exposure

I use ChimeraX to visualise proteins, and will show example commands below. This should be doable, in principle, in any standard protein visualisation app. For ease of visualisation in standard apps, these functions save pdb files where the b-factor column is used to store solvent exposure scores.

#### To color proteins with standard coloring (assuming the model is #1):
```
color byattribute bfactor #1 palette 20,#000000:15,#000088:10,#ff0000:5,#ffff00:0,#ffffff
```

#### To get local resolution by atom in ChimeraX:
Ensure the map and model are correctly aligned!

```
measure mapvalues #3 atoms #1 attribute locres
```

```
save 'XXXX\pdbs\out\XXXX.defattr' attrName locres models #1
```

(assuming model is #1, map is #2; useful for alignment, and local resolution map is #3)

I would love to integrate visualisation at some point in the GUI, when time allows.