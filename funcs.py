import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
import matplotlib.pyplot as plt
import plotly.express as px
import os, math, time, psutil, warnings, plotly
from scipy.spatial.distance import pdist

basedir = os.path.dirname(__file__)
standard_residues = ['LYS', 'LEU', 'THR', 'TYR', 'PRO', 'GLU', 'ASP', 'ILE', 'ALA', 'PHE', 'ARG',
                     'VAL', 'GLN', 'GLY', 'SER', 'TRP', 'CYS', 'HIS', 'ASN', 'MET', 'SEC', 'PYL']
'''3 letter codes for all standard amino acid residues.'''


def yes_no(text: str) -> bool:
    """
    Simple text input for user decided yes or no.

    Args:
        text (str): The message to be displayed.

    Returns:
        (bool): True if user inputs 'y', False if user inputs anything else.
    """
    yn = input(text + ' y/n:')
    if yn == 'y':
        return True
    elif yn == 'n':
        return False
    print('Enter y or n')
    yes_no(text=text)
    

def read_pdb_mmcif(filepath: str, append_heteroatoms: 'function' = None) -> PandasPdb | str:
    """
    Reads pdb or mmcif files and returns a PandasPdb of the contents, as well as the filename.

    Args:
        filepath (str): The path of the file to be read.
        append_heteroatoms (function, optional): If there are heteroatoms in the read file, the output of this function decides if any heteroatoms are added to the chain atoms for later use. If False, any are removed.

    Returns:
        atomic_df,filename (PandasPdb, str): The PandasPdb generated from the read file and the filename (basename) without extention.
    """
    filename = os.path.basename(filepath)

    if filename.rsplit('.',1)[1] == 'gz':
        filename = filename.rsplit('.',1)[0]

    filename, fileext = filename.rsplit('.',1)

    if fileext in ['pdb', 'ent']:
        atomic_df = PandasPdb().read_pdb(filepath)
        atomic_df = atomic_df.get_model(1)
    elif fileext == 'cif':
        atomic_df = PandasMmcif().read_mmcif(filepath)
        atomic_df = atomic_df.convert_to_pandas_pdb()
    else:
        raise ValueError('Wrong file format; allowed file formats are .pdb, .pdb.gz, .ent, .ent.gz, .cif, .cif.gz')
    
    if len(atomic_df.df['HETATM']) > 0:
        if callable(append_heteroatoms):
            yn_func = append_heteroatoms
        else:
            yn_func = yes_no

        if yn_func('Would you like to include heteroatoms so they are used in the solvent exposure calculation?'):    
            atomic_df.df['ATOM'] = pd.concat([atomic_df.df['ATOM'], atomic_df.df['HETATM']], ignore_index = True)
            atomic_df.df['ATOM']['record_name'] = 'ATOM'
            atomic_df.df['HETATM'] = atomic_df.df['HETATM'].iloc[0:0]
        else:
            atomic_df.df['HETATM'] = atomic_df.df['HETATM'].iloc[0:0]
    
    try:
        atomic_df.df['ATOM'] = atomic_df.df['ATOM'].drop('model_id', axis = 1)
    except KeyError:
        pass

    try:
        atomic_df.df['ATOM']['segment_id'] = ''
    except KeyError:
        pass

    try:
        atomic_df.df['ATOM']['element_symbol'] = ''
    except KeyError:
        pass

    return atomic_df, filename


def preprocess_iterate(pdb_path: str, 
               pre_path: str,
               yn: 'function',
               include: list = ['C', 'N', 'O', 'S'], 
               redefine_chains: bool = False) -> str:
    """
    Simple preprocessing of pdb and mmcif files. 
    By default, removes all atoms that do not begin with C, N, O, or S; this normally only leaves carbon, nitrogen, oxygen, sulfur, and selenium for biological molecules.
    Non-standard residue names are flagged, and the user decides whether to include or not in the preprocessed file.

    Args:
        pdb_path (str): The path of the file to be preprocessed (typically pdb or mmcif).
        pre_path (str): The path of the folder inside which the preprocessed pdb will be saved.
        yn (function, optional): For obtaining user input for yes/no questions.
        include (list, optional): If the first letter of an atom's 'atom_name' entry in the pdb/mmcif file is in this list, it will be included in the preprocessed file. If not, it will be removed.
        redefine_chains (bool, optional): If true, each chain will be relabeled, starting with A and going on alphabetically.

    Returns:
        out_path (str): The path of the saved preprocessed pdb.
    """
    
    atomic_df, filename = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=yn)

    temp = atomic_df.df['ATOM']['atom_name'].eq(include[0])

    split_occupancy = []
    added_residues = []
    deleted_residues = []

    if redefine_chains:
        residue_counter = atomic_df.df['ATOM'].iloc[0]['residue_number']
        chain = 65
    else:
        residue_counter = 0

    for i, x in atomic_df.df['ATOM'].iterrows():
        if x['residue_name'] in standard_residues or x['residue_name'] in added_residues:
            if x['atom_name'][0] in include and x['occupancy'] > 0.5:
            
                temp.loc[i] = True

                if x['residue_number'] >= residue_counter and redefine_chains:
                    atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                    residue_counter = x['residue_number']
                elif redefine_chains:
                    chain += 1
                    atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                    residue_counter = x['residue_number']

            if x['atom_name'][0] in include and x['occupancy'] == 0.5:
                if x['residue_number'] >= residue_counter and redefine_chains:
                    atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                    residue_counter = x['residue_number']
                elif redefine_chains:
                    chain += 1
                    atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                    residue_counter = x['residue_number']

                if x['chain_id'] + str(x['residue_number']) + x['atom_name'] not in split_occupancy:
                    temp.loc[i] = True
                    split_occupancy += [x['chain_id'] + str(x['residue_number']) + x['atom_name']]
        elif x['residue_name'] in deleted_residues:
            temp.loc[i] = False
        else:
            if yn(f"Would you like to include residue {x['residue_name']}?"):
                added_residues += [x['residue_name']]
                if x['atom_name'][0] in include and x['occupancy'] > 0.5:
                
                    temp.loc[i] = True

                    if x['residue_number'] >= residue_counter and redefine_chains:
                        atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                        residue_counter = x['residue_number']
                    elif redefine_chains:
                        chain += 1
                        atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                        residue_counter = x['residue_number']

                if x['atom_name'][0] in include and x['occupancy'] == 0.5:
                    if x['residue_number'] >= residue_counter and redefine_chains:
                        atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                        residue_counter = x['residue_number']
                    elif redefine_chains:
                        chain += 1
                        atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                        residue_counter = x['residue_number']

                    if x['chain_id'] + str(x['residue_number']) + x['atom_name'] not in split_occupancy:
                        temp.loc[i] = True
                        split_occupancy += [x['chain_id'] + str(x['residue_number']) + x['atom_name']]
            else:
                deleted_residues += [x['residue_name']]
                temp.loc[i] = False

    atomic_df.df['ATOM'] = atomic_df.df['ATOM'][temp]
    out_path = os.path.join(pre_path, filename+'.pdb')
    atomic_df.to_pdb(out_path)

    return out_path


def preprocess(
    pdb_path: str,
    pre_path: str,
    yn: 'function',
    include: list = ['C', 'N', 'O', 'S'],
    redefine_chains: bool = False) -> str:
    """
    Simple preprocessing of pdb and mmcif files. 
    By default, removes all atoms that do not begin with C, N, O, or S; this normally only leaves carbon, nitrogen, oxygen, sulfur, and selenium for biological molecules.
    Non-standard residue names are flagged, and the user decides whether to include or not in the preprocessed file.

    Args:
        pdb_path (str): The path of the file to be preprocessed (typically pdb or mmcif).
        pre_path (str): The path of the folder inside which the preprocessed pdb will be saved.
        yn (function): For obtaining user input for yes/no questions.
        include (list, optional): If the first letter of an atom's 'atom_name' entry in the pdb/mmcif file is in this list, it will be included in the preprocessed file. If not, it will be removed.
        redefine_chains (bool, optional): If true, each chain will be relabeled, starting with A and going on alphabetically.

    Returns:
        out_path (str): The path of the saved preprocessed pdb.
    """
    atomic_df, filename = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=yn)
    atoms = atomic_df.df["ATOM"].copy()

    # Normalize expected columns exist
    required_cols = ["atom_name", "occupancy", "residue_name", "chain_id", "residue_number"]
    for c in required_cols:
        if c not in atoms.columns:
            raise KeyError(f"Expected column '{c}' in ATOM dataframe")

    # 1) Base mask: first-letter of atom_name is in include AND occupancy > 0.5
    atom_prefix = atoms["atom_name"].astype(str).str[0]
    mask_gt_half = (atom_prefix.isin(include)) & (atoms["occupancy"].astype(float) > 0.5)

    # 2) Handle occupancy == 0.5: include only the first occurrence per (chain_id, residue_number, atom_name)
    half_mask = atoms["occupancy"].astype(float) == 0.5
    if half_mask.any():
        # Build a string key safely (cast to str first)
        key = (
            atoms["chain_id"].astype(str)
            + "_"
            + atoms["residue_number"].astype(str)
            + "_"
            + atoms["atom_name"].astype(str)
        )
        # keep the first occurrence of each key among the half-occupancy rows
        # we want: among rows where occupancy == 0.5, mark True for the first row of each key
        first_half = ~key.duplicated() & half_mask
    else:
        first_half = pd.Series(False, index=atoms.index)

    # combine masks: either >0.5 or first half-occurrence
    keep_mask = mask_gt_half | first_half

    # 3) Non-standard residues: prompt once per residue and include/exclude all atoms of that residue
    std_res = set(standard_residues)
    residue_names = pd.Index(atoms["residue_name"].astype(str).unique())
    nonstandard = [r for r in residue_names if r not in std_res]

    added_residues = set()
    deleted_residues = set()
    for res in nonstandard:
        # ask user once per residue type
        include_res = yn(f"Would you like to include residue {res}?")
        if include_res:
            added_residues.add(res)
        else:
            deleted_residues.add(res)

    # Compose final residue mask
    allowed_residues = std_res.union(added_residues)
    residue_ok = atoms["residue_name"].astype(str).isin(allowed_residues)
    residue_deleted = atoms["residue_name"].astype(str).isin(deleted_residues)

    # atoms must satisfy keep_mask AND be allowed by residue selection (and not explicitly deleted)
    keep_mask &= residue_ok & (~residue_deleted)

    # 4) Optionally redefine chains:
    if redefine_chains:
        # We'll reassign chain IDs so that chain labels start at 'A' and increment when residue_number decreases
        # This replicates "relabel each chain starting with A and going on alphabetically".
        # We iterate over residues only (not per atom), building a mapping from original (chain_id,residue_number)
        # to new chain letters; then map it back to atoms.
        res_index = atoms[["chain_id", "residue_number"]].astype(str)
        # Compose residue-level keys maintaining original order
        res_keys = res_index["chain_id"] + "_" + res_index["residue_number"]
        # We want to order by the original dataframe order and detect when residue_number decreases -> new chain
        # Extract residue_numbers as integers for change detection (preserve order)
        resnums = atoms["residue_number"].astype(int).to_numpy()
        new_chain_letters = []
        current_letter_ord = ord("A")
        last_resnum = None
        # We'll treat a decrease in residue_number as chain boundary (as in original code)
        for rn in resnums:
            if last_resnum is None:
                # first residue -> current letter
                new_chain_letters.append(chr(current_letter_ord))
                last_resnum = rn
                continue
            if rn < last_resnum:
                # new chain
                current_letter_ord += 1
            new_chain_letters.append(chr(current_letter_ord))
            last_resnum = rn
        atoms.loc[:, "chain_id"] = new_chain_letters

    # 5) Apply mask and save the preprocessed pdb
    atomic_df.df["ATOM"] = atoms.loc[keep_mask].copy()
    out_path = os.path.join(pre_path, f"{filename}.pdb")
    atomic_df.to_pdb(out_path)

    return out_path


def f2_cutoff(d: float|np.ndarray, cutoff: float = 50.0, eps: float = np.inf) -> float|np.ndarray:
    """
    Vector-safe (accepts scalars or numpy arrays for faster operation) version of scoring function. 
    - Returns 0 for distances, d, above 0
    - Returns d ** -2 for distances within cutoff
    - Returns inf for distances ≈ 0 -> atom does not count towards its own score
    - Replaces any nan/inf with finite numbers (0.0) for safety

    Args:
        d (scalar or numpy array): Distance between two atoms or array of distances between atoms.
        cutoff (float, optional): Cutoff distance; inputs greater than this will return 0.
        eps (float, optional): Epsilon used for distance ≈ 0. Purposefully infinite so that each atom does not contribute to its own score.

    Returns:
        scores (float or numpy array): Score or array of scores for input of scalar d or numpy array d, respectively.
    """
    da = np.asarray(d, dtype=float)

    # avoid warnings but still produce controlled values
    with np.errstate(divide='ignore', invalid='ignore'):
        # replace zeros with eps before inverting/squaring to avoid inf
        safe = np.where(da == 0.0, eps, da)
        scores = 1.0 / (safe ** 2)

    # zero out above cutoff
    scores = np.where(da > cutoff, 0.0, scores)

    # replace any non-finite values (shouldn't be any now) with 0.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # If input was scalar, return scalar float for compatibility
    if np.isscalar(d):
        return float(scores)
    return scores
  

def exposure_not_vectorizable(pdb_path: str,
                              out_path: str,
                              assignment: None | dict[str, np.ndarray] = None, 
                              funcs: dict[str, 'function'] = {'2c50': f2_cutoff}, 
                              max_scores: dict[str, float] = {'2c50': 26.5}, 
                              save_matrix: bool = False, 
                              save_scores_as_vector: bool = False,
                              print_percentages: list | np.ndarray = [1,5,25,50,75],
                              progress_callback: 'function' = None):
    """
    Slower method of calculating solvent exposure. Will work with scoring functions that are not vectorizable. Also provides updates on progress at user-defined intervals.

    Args:
        pdb_path (str): The path of the file to use in score calculation (typically pdb or mmcif). It has n atoms.
        out_path (str): The path of the folder inside which the output pdb will be saved.
        assignment (None or dict[str, numpy array], optional): Dictionary with value(s) that are length n numpy arrays. 
            As standard, these are boolean arrays used to obtain solvent exposure scores only accounting for the contribution of atoms with entries = True/1 while ignore the contribution of atoms with entries False/0. 
            In principle, this can be used for any algebraic operation, including calculating solvent exposure scores with weighted scores from each atom.
        funcs (dict[str: function], optional): Dictionary with value(s) calling scoring function(s). 
            Keys are used for output file naming, in this case 2c50 meaning d**-2 scoring function with a distance cutoff at 50 Å (the default scoring function).
        max_scores (dict[str: float], optional): Dictionary with value(s) corresponding to the maximum value for the corresponding scoring function(s). 
            The keys used should be those used as the keys in funcs.
        save_matrix (bool, optional): If True, the n x n numpy array giving each atom-atom score is saved in the out_path folder in .npy format.
        save_scores_as_vector (bool, optional): If True, the 1 x n numpy array of calculated solvent exposure scores is saved in the out_path folder in .npy format.
        print_percentages (list[float|int]): A list of values, x, such that when the calculation is x% complete, a progress message containing completion percentage and estimated time remaining is put out.
        progress_callback (None or function, optional): If None, as default, progress messages are printed. If a function is given, custom behaviour can be implemented, such as for printing in a GUI during the run. 

    Returns:
        out (list[list[str, float, float]]): A list with one list entry (sublist) per pdb saved. 
            Each sublist contains three entries: the first is the path of the saved pdb, the second is the minimum solvent exposure score in the saved pdb, and the third is the maximum solvent exposure score in the saved pdb.

    Raises:
        TypeError: assignment must be None or dict.
    """
    atomic_df, filename = read_pdb_mmcif(filepath=pdb_path)

    coords = np.vstack((atomic_df.df['ATOM']['x_coord'].to_numpy(), 
                        atomic_df.df['ATOM']['y_coord'].to_numpy(), 
                        atomic_df.df['ATOM']['z_coord'].to_numpy())).T

    pair_scores = {}

    out = []

    n = len(coords)
    for key in funcs:
        pair_scores[key] = np.zeros((n, n))

    print_i = [round((2*n - 3 - np.sqrt((2*n - 3)**2 - 4*(2-2*n+n*(n-1)*percent/100))) / 2) for percent in print_percentages]

    start = time.time()
    for i, coord1 in enumerate(coords):
        for j, coord2 in enumerate(coords[i+1:]):
            distance = np.linalg.norm(coord1-coord2)
            for key, func in funcs.items():
                pair_scores[key][i,i+j+1] = pair_scores[key][i+j+1,i] = func(distance)
        if i in print_i:
            msg = (f"Exposure calculation {print_percentages[print_i.index(i)]}% complete. "
                   f"Estimated {math.ceil((100 - print_percentages[print_i.index(i)]) * (time.time() - start)/print_percentages[print_i.index(i)])} seconds remaining.")

            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    print(msg)
            else:
                print(msg)




    if assignment == None:
        for key, mat in pair_scores.items():
            temp = np.sum(mat, axis = 0)
            atomic_df.df['ATOM']['b_factor'] = max_scores[key] - temp
            atomic_df.to_pdb(os.path.join(out_path, filename + '_' + key + '.pdb'))

            out += [[os.path.join(out_path, filename + '_' + key + '.pdb'), min(max_scores[key] - temp), max(max_scores[key] - temp)]]
            
            if save_scores_as_vector:
                np.save(os.path.join(out_path, filename + '_' + key + '_vec.npy', temp))

            if save_matrix:
                np.save(os.path.join(out_path, filename + '_' + key + '_mat.npy', mat))
    
    elif type(assignment) == dict:
        mat_not_saved = True
        for k, assigment_vert in assignment.items():
            for key, mat in pair_scores.items():
                temp = assigment_vert @ mat
                atomic_df.df['ATOM']['b_factor'] = max_scores[key] - temp
                atomic_df.to_pdb(os.path.join(out_path, filename + '_' + k + '_' + key + '.pdb'))
                out += [[os.path.join(out_path, filename + '_' + k + '_' + key + '.pdb'), min(max_scores[key] - temp), max(max_scores[key] - temp)]]

                if save_scores_as_vector:
                    np.save(os.path.join(out_path, filename + '_' + k + '_' + key + '_vec.npy'), max_scores[key] - temp)

                if save_matrix and mat_not_saved:
                    np.save(os.path.join(out_path, filename + '_' + key + '_mat.npy'), mat)
                    mat_not_saved = False

    else:
        raise TypeError("assignment must be None or dict")

    return out


def exposure(pdb_path: str,
             out_path: str,
             assignment: None | dict[str, np.ndarray] = None, 
             funcs: dict[str, 'function'] = {'2c50': f2_cutoff}, 
             max_scores: dict[str, float] = {'2c50': 26.5}) -> list[list[str|float|float]]:
    """
    Saves a pdb with b-factor set to the solvent exposure score for all atoms in the pdb supplied. 
    It is recommended to run on preprocessed pdb files, so that decisions on how to handle non-standard atoms and residues are made and hydrogen atoms are removed.
    A standard scoring function and maximum score are used by default, but one in able to experiment with these if they are interested.

    Args:
        pdb_path (str): The path of the file to use in score calculation (typically pdb or mmcif). It has n atoms.
        out_path (str): The path of the folder inside which the output pdb will be saved.
        assignment (None or dict[str, numpy array], optional): Dictionary with value(s) that are length n numpy arrays. 
            As standard, these are boolean arrays used to obtain solvent exposure scores only accounting for the contribution of atoms with entries = True/1 while ignore the contribution of atoms with entries False/0. 
            In principle, this can be used for any algebraic operation, including calculating solvent exposure scores with weighted scores from each atom.
        funcs (dict[str: function], optional): Dictionary with value(s) calling scoring function(s). 
            Keys are used for output file naming, in this case 2c50 meaning d**-2 scoring function with a distance cutoff at 50 Å (the default scoring function).
        max_scores (dict[str: float], optional): Dictionary with value(s) corresponding to the maximum value for the corresponding scoring function(s). 
            The keys used should be those used as the keys in funcs.

    Returns:
        out (list[list[str, float, float]]): A list with one list entry (sublist) per pdb saved. 
            Each sublist contains three entries: the first is the path of the saved pdb, the second is the minimum solvent exposure score in the saved pdb, and the third is the maximum solvent exposure score in the saved pdb.

    Raises:
        TypeError: assignment must be None or dict.
    """
    
    atomic_df, filename = read_pdb_mmcif(filepath=pdb_path)

    coords = np.vstack((atomic_df.df['ATOM']['x_coord'].to_numpy(), 
                        atomic_df.df['ATOM']['y_coord'].to_numpy(), 
                        atomic_df.df['ATOM']['z_coord'].to_numpy())).T
    out = []

    n = coords.shape[0]
    d_cond = pdist(coords)  
    for key, func in funcs.items():             # condensed distances (len m = n*(n-1)/2)
        vals = func(d_cond)                  # elementwise func applied to condensed distances
        # accumulate into per-atom sum vector without expanding to n x n
        if assignment == None:
            sums = np.zeros(n, dtype=vals.dtype)
            idx = 0
            for i in range(n-1):
                # length of this row in condensed form = n - i - 1
                l = n - i - 1
                sums[i] += vals[idx: idx + l].sum()
                # the vals in this block correspond to pairs (i, i+1..n-1)
                sums[i+1: n] += vals[idx: idx + l]   # vector add to the other atoms
                idx += l
                
            atomic_df.df['ATOM']['b_factor'] = max_scores[key] - sums
            atomic_df.to_pdb(os.path.join(out_path, filename + '_' + key + '.pdb'))

            out += [[os.path.join(out_path, filename + '_' + key + '.pdb'), min(max_scores[key] - sums), max(max_scores[key] - sums)]]#, elapsed, len(coords)]]
    
        elif type(assignment) == dict:
            for k, assignment_vert in assignment.items():
                idx = 0
                sums = np.zeros(n, dtype=vals.dtype)
                for i in range(n - 1):
                    l = n - i - 1
                    block = vals[idx: idx + l]          # values for pairs (i, i+1..n-1)
                    # contribution to sums[i] from j>i: sum_j v_ij * assignment[j]
                    sums[i] += np.dot(block, assignment_vert[i+1: n])
                    # contribution to sums[j] from i when assignment[i] == 1:
                    if assignment_vert[i] != 0:
                        # we need to add v_ij * assignment[i] to sums[j] for j>i
                        sums[i+1: n] += block * assignment_vert[i]
                    idx += l

                # elapsed = time.time()-start
                atomic_df.df['ATOM']['b_factor'] = max_scores[key] - sums
                atomic_df.to_pdb(os.path.join(out_path, filename + '_' + k + '_' + key + '.pdb'))
                out += [[os.path.join(out_path, filename + '_' + k + '_' + key + '.pdb'), min(max_scores[key] - sums), max(max_scores[key] - sums)]]#, elapsed, len(coords)]]

        else:
            raise TypeError("assignment must be None or dict")
    return out


def average_score(filepath: str, backbone: bool = False) -> list[list[str|float|float]]:
    """
    Averages the solvent exposure scores or local resolution values within a pdb file or defattr file, respectively.
    For each input file, one or two files will be returned: 
    - One where the value of every atom within a residue is set to the average of all atoms within that residue (filename ends with _avgbyres)
    - One (if backbone is True) where the value of every atom within a residue is set to the average of all backbone atoms within that residue (filename ends with _avgbyresbb)

    Args:
        filepath (str): The path of the file to have values averaged (typically pdb or defattr).
        backbone (bool, optional): If True, an additional file will be returned such that the value of every atom within a residue is set to the average of all backbone atoms within that residue.
        
    Returns:
        out (list[list[str, float, float]]): A list with one list entry (sublist) per file saved. 
            Each sublist contains three entries: the first is the path of the saved file, the second is the minimum solvent exposure score or local resolution value in the saved file, and the third is the maximum solvent exposure score or local resolution value in the saved file.
    """        
    out = []
    if filepath[-7:] == 'defattr':
        localres = pd.read_csv(filepath, sep = '\t', header = 3, usecols = [1,2], names = ['atom', 'localres']).set_index('atom')

        current_residue = localres.iloc[0].name.split('@')[0]

        scores = np.zeros(len(localres))
        backbone_scores = np.zeros(len(localres))

        atom_count = 0
        score = 0
        backbone_atom_count = 0
        backbone_score = 0
        i=0

        for ind, res in localres.iterrows():
            if ind.split('@')[0] == current_residue:
                atom_count+=1
                score+=res['localres']
                if ind.split('@')[1] in ['C', 'N', 'O', 'CA']:
                    backbone_atom_count+=1
                    backbone_score+=res['localres']
            else:
                for j in range(atom_count):
                    scores[i-j-1] = score / atom_count
                    if backbone_atom_count != 0:
                        backbone_scores[i-j-1] = backbone_score / backbone_atom_count

                atom_count=1
                score=res['localres']
                current_residue = ind.split('@')[0]

                if ind.split('@')[1] in ['C', 'N', 'O', 'CA']:
                    backbone_atom_count=1
                    backbone_score=res['localres']
                else:
                    backbone_atom_count=0
                    backbone_score=0
            i+=1

        score_df = pd.read_csv(filepath, sep = '\t', header = 3, names = ['', 'atom', 'localres'])
        score_df['localres'] = scores
        score_df.to_csv(filepath[:-8] + '_avgbyres.defattr', sep = '\t', header=['attribute: locres \nrecipient: atoms \nmatch mode: 1-to-1', '', ''], index=False)
        out += [[filepath[:-8] + '_avgbyres.defattr', min(scores), max(scores)]]

        if backbone:
            score_df['localres'] = backbone_scores
            score_df.to_csv(filepath[:-8] + '_avgbyresbb.defattr', sep = '\t', header=['attribute: locres \nrecipient: atoms \nmatch mode: 1-to-1', '', ''], index=False)
            out += [[filepath[:-8] + '_avgbyresbb.defattr', min(backbone_scores), max(backbone_scores)]]

    else:
        atomic_df, filename = read_pdb_mmcif(filepath=filepath)

        current_residue = atomic_df.df['ATOM'].iloc[0].loc['chain_id'] + str(atomic_df.df['ATOM'].iloc[0].loc['residue_number'])

        scores = np.zeros(len(atomic_df.df['ATOM']))
        backbone_scores = np.zeros(len(atomic_df.df['ATOM']))

        atom_count = 0
        score = 0
        backbone_atom_count = 0
        backbone_score = 0

        for i, x in atomic_df.df['ATOM'].iterrows():
            if x['chain_id'] + str(x['residue_number']) == current_residue:
                atom_count+=1
                score+=x['b_factor']
                if x['atom_name'] in ['C', 'N', 'O', 'CA']:
                    backbone_atom_count+=1
                    backbone_score+=x['b_factor']
                if x.equals(atomic_df.df['ATOM'].iloc[-1]):
                    for j in range(atom_count):
                        scores[i-j] = score / atom_count
                        if backbone_atom_count != 0:
                            backbone_scores[i-j] = backbone_score / backbone_atom_count

                    atom_count=1
                    score=x['b_factor']

            else:
                for j in range(atom_count):
                    scores[i-j-1] = score / atom_count
                    if backbone_atom_count != 0:
                        backbone_scores[i-j-1] = backbone_score / backbone_atom_count

                atom_count=1
                score=x['b_factor']
                current_residue = x['chain_id'] + str(x['residue_number'])

                if x['atom_name'] in ['C', 'N', 'O', 'CA']:
                    backbone_atom_count=1
                    backbone_score=x['b_factor']
                else:
                    backbone_atom_count=0
                    backbone_score=0

                if i == len(atomic_df.df['ATOM']) - 1:
                    scores[i] = score
                    backbone_scores[i] = score

        atomic_df.df['ATOM']['b_factor'] = scores
        atomic_df.to_pdb(filepath.split('.',1)[0] + '_avgbyres.pdb')
        out += [[filepath.rsplit('.',1)[0] + '_avgbyres.pdb', min(scores), max(scores)]]

        if backbone:
            atomic_df.df['ATOM']['b_factor'] = backbone_scores
            atomic_df.to_pdb(filepath.split('.',1)[0] + '_avgbyresbb.pdb')
            out += [[filepath.rsplit('.',1)[0] + '_avgbyresbb.pdb', min(backbone_scores), max(backbone_scores)]]

    return out
    

def create_3_vectors(pdb_path: str, chain1: str | list, feature: str) -> dict[str, np.ndarray]:      
    """
    Creates three assignment vectors:
    - One with values True/1 for all atoms that are/are in chain1 and False/0 for all that are not.
    - One with values False/0 for all atoms that are/are in chain1 and True/1 for all that are not.
    - One with values True/1 for all atoms. This array will output the same solvent exposure scores as running exposure with assignment = None.
    This is useful for understanding the contribution to solvent exposure score from some atoms, such as those in adduct/detergent molecules. For example:
    - If all detergent molecules have their own chain_id, then feature can be set to chain_id, and chain1 set to that chain_id, so that the solvent exposure can be calculated for contributions solely from the detergent molecules, solely from the protein, and when including every atom.
    - If the detergent/adduct molecules do not have their own chain_id, but are labeled as something else under residue_name, this can be used for calculating the contributions.
    If you require multiple entries in chain1, I recommend assigning chain1 the list of entries that is the shorter of your two options, to keep filenames shorter.

    Args:
        pdb_path (str): The path of the pdb file for which vectors will be generated (and subsequently calculations will be run).
        chain1 (str or list): The entry or entries of feature to be included in the first assignment vector.
        feature (str): The identifier used to separate the atoms of interest. Most commonly used: chain_id and residue_name.

    Returns:
        (dict[str: numpy array]): A dict with three key-value pairs, one per assignment vector. 
            The key, a string, is used for future file naming and keeping track of which entries are in assignment vector 1, and therefore those that aren't in assignment vector 2. The key for the True/1 vector is tot.
    """    
    atomic_df, _ = read_pdb_mmcif(filepath=pdb_path)

    out_tot = np.ones(len(atomic_df.df['ATOM']), dtype=bool)

    if type(chain1) == str:
        out1 = np.array(atomic_df.df['ATOM'][feature].eq(chain1)).astype(bool)
        name = chain1
    elif type(chain1) == list:
        for ind, chain in enumerate(chain1):
            if ind == 0:
                temp = atomic_df.df['ATOM'][feature].eq(chain)
                name = chain
            else:
                name = name + '_' + chain
                temp = temp | atomic_df.df['ATOM'][feature].eq(chain)
        out1 = np.array(temp).astype(bool)

    out2 = np.invert(out1)

    return {name: out1, 'not'+name: out2, 'tot': out_tot}


def create_vectors(pdb_path: str, include: str | list, feature: str) -> dict[str, np.ndarray]:
    """
    Creates one assignment vector, with values True/1 for all atoms that have entries in include for feature and False/0 for all that are not.
    This is useful for understanding the contribution to solvent exposure score from some atoms, such as those in adduct/detergent molecules.

    Args:
        pdb_path (str): The path of the pdb file for which vectors will be generated (and subsequently calculations will be run).
        include (str or list): The entry or entries of feature to be included in the first assignment vector.
        feature (str): The identifier used to separate the atoms of interest. Most commonly used: chain_id and residue_name.

    Returns:
        out (dict[str: numpy array]): A dict with one key-value pair. The name of what is included, as a string, is the key. The value is the assignment vector.
    """    
    atomic_df, _ = read_pdb_mmcif(filepath=pdb_path)

    if type(include) == str:
        return {include: np.array(atomic_df.df['ATOM'][feature].eq(include)).astype(bool)}
    elif type(include) == list:
        for ind, chain in enumerate(include):
            if ind == 0:
                temp = atomic_df.df['ATOM'][feature].eq(chain)
                name = chain
            else:
                name = name + '_' + chain
                temp = temp | atomic_df.df['ATOM'][feature].eq(chain)
        return {name: temp}
    raise TypeError('Incorrect type for include. Must be string or list.')


def score_v_localres(pdb_path: str, 
                     defattr_path: str, 
                     only_chain: bool | list[str] = False,
                     called_by_GUI: bool = False, 
                     backboneonly: bool = False, 
                     inverse: bool = True, 
                     interactive: bool = False) -> dict:
    """
    This function plots solvent exposure score per atom versus the local resolution at that atom. 
    NOTE: this is only valuable information for gas-phase dehydration if the local resolution at that atom is obtained using a confidently assigned model, aligned properly within a cryo-EM map (and the local resolution map/mask)!

    Args:
        pdb_path (str): The path of the pdb file with solvent exposure scores (saved as b-factor entries).
        defattr_path (str): The path of the defattr file with local resolution values. These are generated in ChimeraX with these commands, where #1 is the aligned atomic model and #3 is the local resolution map:
            measure mapvalues #3 atoms #1 attribute locres
            save 'XXXX\\pdbs\\out\\XXXX.defattr' attrName locres models #1
        only_chain (bool or list[str], optional): If False, as default, all atoms (for which there are both solvent exposure scores and local resolution values) are plotted. 
            If a list of strings is provided, only atoms with chain_id that are in the list are plotted. This is useful for proteins with non-crystallographic symmetry, as only one asymmetric subunit needs to be plotted.
        called_by_GUI (bool, optional): If False, as default, the function plots using matplotlib. When True, the function does not actually plot, but returns values to be used elsewhere for plotting.
        backboneonly: (bool, optional): If False, as default, all atoms (for which there are both solvent exposure scores and local resolution values) are plotted.
            If True, only backbone atoms are plotted.
        inverse (bool, optional): If True, as default, the y-values are plotted as 1 / (Local Resolution). If false, they are plotted as Local Resolution.
        interactive (bool, optional): If False, as default, a static plot is shown. If True, the plot is interactive, with annotations popping up over points where the mouse is hovering. 
            This can be rather inconsistent to get working, but is currently working consistently in the GUI, which I recommend using for interactive plotting purposes.

    Returns:
        (dict): If called_by_GUI, a dict is returned containing key (str)-value pairs of:
            - 'x': numpy array of x values,
            - 'y': numpy array of y values,
            - 'names': list of point names (list[str]),
            - 'inverse': inverse (bool),
            - 'xlabel': x-axis label for plotting (str),
            - 'ylabel': y-axis label for plotting (str).
            If not called_by_GUI, a dict is returned containing key (str)-value pairs of:
            - 'fig': plot fig (Figure), 
            - 'ax': plot ax (Axes), 
            - 'sc': scatterplot sc (PathCollection), 
            - 'names': list of point names (list[str]).
    """
    if called_by_GUI:
        localres = pd.read_csv(defattr_path, sep = '\t', header = 3, usecols = [1,2], names = ['atom', 'localres']).set_index('atom')

        out = np.zeros((len(localres), 2))
        names = list(np.zeros(len(localres)).astype(int).astype(str))

        fileext = pdb_path.split('.',1)[1]
        if fileext in ['pdb', 'pdb.gz', 'ent', 'ent.gz']:
            atomic_df = PandasPdb().read_pdb(pdb_path)
            atomic_df = atomic_df.get_model(1)
            
        elif fileext in ['cif', 'cif.gz']:
            atomic_df = PandasMmcif().read_mmcif(pdb_path)
            atomic_df = atomic_df.convert_to_pandas_pdb()
        else:
            raise ValueError('Wrong file format; allowed file formats are .pdb, .pdb.gz, .ent, .ent.gz, .cif, .cif.gz')

        try:
            atomic_df.df['ATOM'] = atomic_df.df['ATOM'].drop('model_id', axis = 1)
        except KeyError:
            pass
        df = atomic_df.df['ATOM'].set_index(['chain_id','residue_number', 'atom_name'])

        i = 0
        errorcount = 0

        if localres.iloc[0].name[0] == '#':
            k = len(localres.iloc[0].name.split('/')[0])
        else:
            k = 0

        for ind, row in localres.iterrows():
            if not backboneonly or ind.split('@')[1] in ['C', 'N', 'O', 'CA']:
                if only_chain:
                    if ind[1+k:2+k] in only_chain:
                        try:
                            if type(df.loc[(ind[1+k:2+k], int(ind[3+k:].split('@')[0])), 'b_factor'][ind[3+k:].split('@')[1]]) == pd.core.series.Series:
                                errorcount+=1
                            else:
                                out[i,0] = df.loc[(ind[1+k:2+k], int(ind[3+k:].split('@')[0])), 'b_factor'][ind[3+k:].split('@')[1]]
                                out[i,1] = row['localres']
                                names[i] = ind
                                i+=1
                        except KeyError:
                            errorcount+=1
                    else:
                        errorcount+=1

                else:
                    try:
                        if type(df.loc[(ind[1+k:2+k], int(ind[3+k:].split('@')[0])), 'b_factor'][ind[3+k:].split('@')[1]]) == pd.core.series.Series:
                            errorcount+=1
                        else:
                            out[i,0] = df.loc[(ind[1+k:2+k], int(ind[3+k:].split('@')[0])), 'b_factor'][ind[3+k:].split('@')[1]]
                            out[i,1] = row['localres']
                            names[i] = ind
                            i+=1
                    except KeyError:
                        errorcount+=1
            else:
                errorcount += 1

        if errorcount != 0:
            out = out[:-errorcount].T
            names = names[:-errorcount]
        else:
            out = out.T

        if inverse:
            x = out[0]
            y = 1.0 / out[1]
            ylabel = 'Local Resolution at Atom in Model (1/Å)'
        else:
            x = out[0]
            y = out[1]
            ylabel = 'Local Resolution at Atom in Model (Å)'

        xlabel = 'Atom Exposure Score (Arbitrary Units)'

        return {
            'x': x,
            'y': y,
            'names': names,
            'inverse': inverse,
            'xlabel': xlabel,
            'ylabel': ylabel
        }

    else:
        plt.close()
        
        fig,ax = plt.subplots()

        localres = pd.read_csv(defattr_path, sep = '\t', header = 3, usecols = [1,2], names = ['atom', 'localres']).set_index('atom')

        out = np.zeros((len(localres), 2))

        names = list(np.zeros(len(localres)).astype(int).astype(str))

        atomic_df, _ = read_pdb_mmcif(filepath=pdb_path)

        df = atomic_df.df['ATOM'].set_index(['chain_id','residue_number', 'atom_name'])

        i=0

        errorcount = 0

        if localres.iloc[0].name[0] == '#':
            k = len( localres.iloc[0].name.split('/')[0] )
        else:
            k = 0

        for ind, row in localres.iterrows():
            if not backboneonly or ind.split('@')[1] in ['C', 'N', 'O', 'CA']:
                try:
                    if type(df.loc[(ind[1+k:2+k], int(ind[3+k:].split('@')[0])), 'b_factor'][ind[3+k:].split('@')[1]]) == pd.core.series.Series:
                        errorcount+=1
                    else:
                        out[i,0] = df.loc[(ind[1+k:2+k], int(ind[3+k:].split('@')[0])), 'b_factor'][ind[3+k:].split('@')[1]]
                        out[i,1] = row['localres']
                        names[i] = ind
                        i+=1
                except KeyError:
                    errorcount+=1
            else:
                errorcount+=1

        if errorcount != 0:
            out = out[:-errorcount].T
            names = names[:-errorcount]
        else:
            out=out.T

        if inverse:
            sc = plt.scatter(out[0], 1/out[1], s=3)#, color=(0,0,1,0.5))
            # plt.plot(np.unique(out[0]), np.poly1d(np.polyfit(out[0], 1/out[1], 1))(np.unique(out[0])), color = (0,0,0))
            plt.ylabel('Local Resolution at Atom in Model (1/Å)')
        else:
            sc = plt.scatter(out[0], out[1], s=3)#, color=(0,0,1,0.5))
            plt.ylabel('Local Resolution at Atom in Model (Å)')
        plt.xlabel('Atom Exposure Score (Arbitrary Units)')

        ymin, ymax = ax.get_ylim()
        ticks = reciprocal_ticks(ymin, ymax)
        ax.set_yticks(1/ticks)
        ax.set_yticklabels(f'1/{tick:.2g}' for tick in ticks)

        if interactive:
            annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)
            
            def update_annot(ind):
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
                annot.set_text(text)
                # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
                annot.get_bbox_patch().set_alpha(0.4)
                
            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", hover)

            # Avoid calling plt.show() if a Qt application is running (prevents stealing focus / spawning windows).
            try:
                # import lazily so funcs.py doesn't require PyQt unless running under GUI
                from PyQt6.QtWidgets import QApplication
                qt_app = QApplication.instance()
            except Exception:
                qt_app = None

            if qt_app is None:
                # no Qt app -> safe to show normally (scripts / notebooks)
                try:
                    plt.show()
                except Exception:
                    plt.ion()
                    plt.show()
            else:
                # running inside Qt app -> do not block; optionally flush drawings
                try:
                    plt.pause(0.001)
                except Exception:
                    pass
            return {'fig': fig, 'ax': ax, 'sc': sc, 'names': names}
        else:
            # Avoid calling plt.show() if a Qt application is running (prevents stealing focus / spawning windows).
            try:
                # import lazily so funcs.py doesn't require PyQt unless running under GUI
                from PyQt6.QtWidgets import QApplication
                qt_app = QApplication.instance()
            except Exception:
                qt_app = None

            if qt_app is None:
                # no Qt app -> safe to show normally (scripts / notebooks)
                try:
                    plt.show()
                except Exception:
                    plt.ion()
                    plt.show()
            else:
                # running inside Qt app -> do not block; optionally flush drawings
                try:
                    plt.pause(0.001)
                except Exception:
                    pass
            return {'fig': fig, 'ax': ax, 'sc': sc, 'names': names}


def visualize(pdb_path: str,
              b_factor_range: list = [0, 20],
              append_heteroatoms: 'function' = yes_no) -> 'plotly.graph_objs._figure.Figure':
    """
    Build and return a Plotly Figure for the given pdb_path.
    DOES NOT call fig.show() so GUI can embed the result.
    """
    # defensive: validate pdb_path
    if not pdb_path:
        raise ValueError("No pdb_path provided to visualize()")
    # optional: allow absolute/relative paths; don't hardcode a file
    atomic_df, _ = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=append_heteroatoms)

    fig = px.scatter_3d(
        atomic_df.df["ATOM"],
        x='x_coord', y='y_coord', z='z_coord',
        color='b_factor',
        color_continuous_scale=[
            (0, '#ffffff'),
            (0.25, '#ffff00'),
            (0.5, '#ff0000'),
            (0.75, '#000088'),
            (1, '#000000')
        ],
        range_color=b_factor_range
    )
    fig.update_traces(marker_size=4)

    return fig


def features(pdb_path: str, feature: str, yn: 'function' = yes_no) -> list:
    """
    Creates a list of all unique entries within a pdb or mmcif file for an individual feature/identifier (e.g., chain_id, residue_name, atom_name).

    Args:
        pdb_path (str): The path of the pdb (or mmcif) file.
        feature (str): The identifier for which to return all unique entries. Do not use mmcif identifier strings, as the file is converted to pdb before the output list is generated.
        yn (function, optional): For obtaining user input for yes/no questions.

    Returns:
        out (list): A list of all unique entries under feature. Typically a list of strings, integers, or floats.
    """    
    atomic_df, _ = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=yn)
    return atomic_df.df['ATOM'][feature].drop_duplicates().tolist()


def reciprocal_ticks(mn: float,
                     mx: float, 
                     n: int|float = 4, 
                     intervals: list|np.ndarray = [1,2,5,10]) -> np.ndarray:
    """
    Dynamically generates ticks for plotting inverse/reciprocal y values, such that 1/y is evenly spaced, not the ticks.

    Args:
        mn (float): minimum y value (1/y is greatest)
        mx (float): maximum y value (1/y is smallest)
        n (int | float, optional): If fewer ticks than n will be displayed, a tighter interval is chosen, resulting in more ticks.
        intervals (list or numpy array, optional): List of values (typically integers but floats may be necessary for other y ranges) corresponding to number of ticks per 1/y to 1/(y+1) step. For example:
            - At interval 1, ticks are shown at 1/1, 1/2, 1/3, 1/4, etc.
            - At interval 5, ticks are shown at 1/1, 1/1.2, 1/1.4, 1/1.6, etc.

    Returns:
        ticks (numpy array): Array of all y values of ticks to be plotted.

    Raises:
        ValueError: If mn and/or mx values are negative.

    Warnings:
        Warned if mn equals mx.
    """
    if mn <= 0:
        raise ValueError('All y values must be positive')
    if mx <= 0:
        raise ValueError('All y values must be positive')
    if mn == mx:
        ticks = mn
        warnings.warn('Minimum value is equal to maximum value. Only tick returned is this value.')
    else:
        ticks = []
        for interval in intervals:
            if interval/mn - interval/mx < n:
                continue
            else:
                tick = math.ceil(interval/mn)/interval
                while tick > interval/mx:
                    ticks += [tick]
                    tick += -1/interval
                break

    return np.array(ticks)


def max_n_for_full_matrix(fraction_of_avail: float = 0.5, dtype: type = np.float64) -> int:
    """
    Calculates size, n, of an n x n matrix with specified dtype that can be generated when using a specified fraction of available memory.

    Args:
        fraction_of_avail (float, optional): Maximum fraction of available memory to use for the matrix.
        dtype (type): data type of matrix elements.

    Returns:
        (int): Size, n, of the largest n x n matrix to use, at maximum, the fraction of availale memory.
    """
    # fraction_of_avail: fraction of available RAM to use for the matrix
    avail = psutil.virtual_memory().available
    bytes_per_element = np.dtype(dtype).itemsize
    max_bytes = int(avail * fraction_of_avail)
    # n^2 * bytes_per_element <= max_bytes  ->  n <= sqrt(max_bytes/bytes_per_element)
    return int(math.floor(math.sqrt(max_bytes / bytes_per_element)))

