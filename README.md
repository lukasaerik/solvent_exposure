# Solvent Exposure

Simple tools to handle pdbs/mmcifs and calculate solvent exposure scores

## Installation
### In python

I recommend running this in its own environment. I will give examples for Anaconda/Miniconda, but use your own systems and pip if preferred.

```python
conda create -n solvent_exposure matplotlib scipy pyqt plotly
```

PyQt is not strictly necessary for only running in command line, as .py files, or in a notebook (.ipynb files). Feel free to omit if preferred

Activate the environment and install the other dependencies:
```python
conda activate solvent_exposure
```

```python
conda install -c conda-forge biopandas mplcursors pyqtwebengine
```

psutil is not always installed by default, install if necessary
```python
conda install -c conda-forge psutil
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

I will not be making a standalone app available until major changes to the GUI and underlying functions are finished.

## Running
```funcs.py``` contains all standard functions, and is called by ```app.py``` and ```solvent_exposure.ipynb``` for running in a GUI or notebook, respectively. These are now annotated rather verbosely, so read the docstrings for more clarification.

Running ```app.py``` generates a GUI. This is the easiest way to use these functions as intended for most users. There are only limited options for custom calculations, but the GUI should cover all standard use cases. Modifying the code slightly to include custom scoring functions is quite straightforward.

Under settings, you can decide whether to weight the contributions to exposure score from all atoms by their atomic mass. When doing so, it is important to remember your raw score will change by more than an order of magnitude. A proper maximum score must be calculated for both weighte (by atomic masses) and unweighted calculations. This can now be done in the GUI under Manual -> Exposure Calculation or using ```funcs.max_exposure_score()```. Currently, the maximum score is given by the mean score plus two (n_sigmas) times the standard deviation of scores, but this can be changed if you so wish.

```solvent_exposure.ipynb``` is a notebook version that shows how to run standard calculations as intended, and a couple advanced options as examples.

```cif_handling.py``` contains the background functions necessary to read, write, and manipulate mmCIF (.cif) files.

As standard, raw (input) pdb/cif files can be placed into ```/pdbs/in/``` and calculated defattr files can be placed into ```/pdbs/out/defattr/```. Preprocessed files are automatically saved to ```/pdbs/preprocessed/``` and files where solvent exposure has been calculated are saved to ```/pdbs/out/```. Standards used for calculating maximum scores are saved under ```/standards/```.

## Visualising solvent exposure

Simple visualisation of the proteins, colored by solvent exposure score, is available in the GUI or using Plotly.

For more options, I use ChimeraX and will show example commands below. This should be doable, in principle, in any standard protein visualisation app. For ease of visualisation in standard apps, these functions save pdb files where the b-factor column is used to store solvent exposure scores.

#### To color proteins with standard coloring (assuming the model is #1):
```
color byattribute bfactor #1 palette 100,#000000:75,#000088:50,#ff0000:25,#ffff00:0,#ffffff
```

A key can be generated with
```
key #ffffff:0 #ffff00:25 #ff0000:50 #000088:75 #000000:100 ticks true
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