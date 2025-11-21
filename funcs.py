import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os, math, time, psutil, warnings, plotly, pickle
from scipy.spatial.distance import cdist

from cif_handling import read_raw_cif, split_header_blocks_footer, parse_loops_from_block_with_offsets, loop_to_dataframe, infer_start_columns, infer_decimal_places, write_loop_from_df_aligned, replace_loop_in_block_text, write_cif_from_parts, canonical_atom_site_order, compute_start_cols_standard_first, canonicalize_atom_site_columns

basedir = os.path.dirname(__file__)
standard_residues = ['LYS', 'LEU', 'THR', 'TYR', 'PRO', 'GLU', 'ASP', 'ILE', 'ALA', 'PHE', 'ARG',
                     'VAL', 'GLN', 'GLY', 'SER', 'TRP', 'CYS', 'HIS', 'ASN', 'MET', 'SEC', 'PYL']
'''3 letter codes for all standard amino acid residues.'''
Cs = ['C', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'CT', 'CM', 'CQ', 'CZQ', 'CME',
      'CA1', 'CA2', 'CB1', 'CB2', 'CD1', 'CD2', 'CD3', 'CE1', 'CE2', 'CE3', 'CE4', 'CE5', 'CE6', 'CG1', 'CG2', 'CG3', 'CH1', 'CH2', 'CH3', 'CZ1', 'CZ2', 'CZ3', 'CT1', 'CT2', 'CT3',
      'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
      "C1'", "C2'", "C3'", "C4'", "C5'",]
Ns = ['N', 'NA', 'NB', 'NC', 'ND', 'NE', 'NZ', 'NT',
      'ND1', 'ND2', 'ND3', 'NE1', 'NE2', 'NH1', 'NH2', 'NT1', 'NT2', 'NT3',
      'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N1+', 'N2+',]
Os = ['O', 'OB', 'OG', 'OH', 'OW', 'OXT',
      'OC1', 'OC2', 'OD1', 'OD2', 'OE1', 'OE2', 'OG1', 'OG2', 'OH2', 'OP1', 'OP2', 'OP3', 'OT1', 'OT2', 'OXT1', 'OXT2',
      'O1', 'O2', 'O3', 'O4', 'O5', 'O6',
      "O2'", "O3'", "O4'", "O5'",
      'O1A', 'O1B', 'O1G', 'O2A', 'O2B', 'O2G', 'O3A', 'O3B', 'O3G',]
Ss = ['S', 'SD', 'SG', 'S1', 'SM', 'S1P', 'S2', 'S3', 'SD1', 'SD2',]
Ps = ['P', 'PA', 'PB', 'PG', 'PC', 'PD', 'PE', 'PG1', 'PG2',]
NAs= ['NA']
MGs= ['MG']
ZNs= ['ZN']
FEs= ['FE', 'FE1', 'FE2', 'FE3']
MNs= ['MN', 'MN1', 'MN2']
CLs= ['CL']
Ks = ['K']
CAs= ['CA']  # note: also alpha-carbon; distinguish by residue name
COs= ['CO', 'CO1']
CUs= ['CU', 'CU1', 'CU2']
HGs= ['HG']
NIs= ['NI']
SEs= ['SE']
BRs= ['BR']
Is = ['I']
SRs= ['SR']
BAs= ['BA']
standard_atoms = Cs + Ns + Os + Ss + Ps
all_atoms = standard_atoms + NAs + MGs + ZNs + FEs + MNs + CLs + Ks + CAs + COs + CUs + HGs + NIs + SEs + BRs + Is + SRs + BAs 


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
    

def cif_to_df(path:str):
    raw = read_raw_cif(path=path)
    header, blocks, footer = split_header_blocks_footer(raw)
    block_name = next(iter(blocks.keys()))
    block_text = blocks[block_name]
    loops = parse_loops_from_block_with_offsets(block_text)

    # 2) find atom_site loop
    atom_loop = None
    for L in loops:
        if any('_atom_site.' in t for t in L['tags']):
            atom_loop = L
            break
    assert atom_loop is not None

    # 3) convert to df (this keeps integers as Int64, floats as Float64)
    df = loop_to_dataframe(atom_loop)

    canonical_order = canonical_atom_site_order()

    df_canon = canonicalize_atom_site_columns(df)

    # 4) infer start columns and decimals (optional, writer will infer for you if you skip)
    start_cols = infer_start_columns(atom_loop)
    decimals_map = infer_decimal_places(atom_loop, df)

    # You asked for specific known starts as example:
    # If you prefer to override or supply exact starts:

    start_cols_override = {
        '_atom_site.id': 8,
        '_atom_site.type_symbol': 17,
        '_atom_site.label_atom_id': 20,
        '_atom_site.label_alt_id': 28,
        '_atom_site.label_comp_id': 30,
        '_atom_site.label_asym_id': 35,
        '_atom_site.label_entity_id': 39,
        '_atom_site.label_seq_id': 44,
        '_atom_site.pdbx_PDB_ins_code': 49,
        '_atom_site.Cartn_x': 51,
        '_atom_site.Cartn_y': 60,
        '_atom_site.Cartn_z': 69,
        '_atom_site.occupancy': 78,
        '_atom_site.B_iso_or_equiv': 83,
        '_atom_site.pdbx_formal_charge': 91,
        '_atom_site.auth_seq_id': 93,
        '_atom_site.auth_comp_id': 99,
        '_atom_site.auth_asym_id': 105,
        '_atom_site.auth_atom_id': 109,
        '_atom_site.pdbx_PDB_model_num': 117
    }

    # merge inferred with overrides (override wins)
    new_starts = compute_start_cols_standard_first(start_cols, df_canon.columns.tolist(), df_canon,
                                                   min_gap=2, default_width=6, 
                                                   start_cols_override=start_cols_override)

    cifdata = header, blocks, footer, block_name, block_text, atom_loop, new_starts, decimals_map
    return df_canon, cifdata


def df_to_cif(outpath, df, cifdata):
    header, blocks, footer, block_name, block_text, atom_loop, new_starts, decimals = cifdata
    # 6) write aligned loop text
    new_loop_text = write_loop_from_df_aligned(
        df=df,
        loop_info=atom_loop,
        tag_order=df.columns.tolist(),
        start_cols=new_starts,       # optional: pass explicit mapping or let it be inferred
        decimals_map=decimals,       # optional: pass explicit mapping or let it be inferred
        float_fmt_template=None,     # use default '{:.{p}f}' behaviour inside function
        missing_token='?',
        indent=''
    )

    # 7) replace exactly using stored offsets and write file back
    assert block_text[atom_loop['raw_start']:atom_loop['raw_end']] == atom_loop['raw_text']
    new_block_text = replace_loop_in_block_text(block_text, atom_loop, new_loop_text)
    blocks[block_name] = new_block_text
    write_cif_from_parts(header, blocks, footer, outpath=outpath)


def read_pdb_mmcif(filepath: str, append_heteroatoms: 'function' = None) -> PandasPdb | str:
    """
    Reads pdb, mmcif, or properly generated pkl files and returns a PandasPdb or dataframe of the contents, as well as the filename and cifdata.

    Args:
        filepath (str): The path of the file to be read.
        append_heteroatoms (function, optional): If there are heteroatoms in the read file, the output of this function decides if any heteroatoms are added to the chain atoms for later use. If False, any are removed.

    Returns:
        atomic_df,filename,cifdata (PandasPdb, str): The PandasPdb generated from the read file and the filename (basename) without extention.
    """

    filename = os.path.basename(filepath)

    if filename.rsplit('.',1)[1] == 'gz':
        filename = filename.rsplit('.',1)[0]

    filename, fileext = filename.rsplit('.',1)

    if fileext in ['pdb', 'ent']:
        atomic_df = PandasPdb().read_pdb(filepath)
        atomic_df = atomic_df.get_model(1)

        if len(atomic_df.df['ANISOU']) > 0:
            if callable(append_heteroatoms):
                yn_func = append_heteroatoms
            elif append_heteroatoms == True:
                atomic_df.df['ANISOU'] = atomic_df.df['ANISOU'].iloc[0:0]
            else:
                yn_func = yes_no
            if append_heteroatoms != True:
                if yn_func('Would you like to delete ANISOU entries?\nGenerally recommended.'):
                    atomic_df.df['ANISOU'] = atomic_df.df['ANISOU'].iloc[0:0]

        if len(atomic_df.df['HETATM']) > 0:
            if append_heteroatoms == True:
                atomic_df.df['ATOM'] = pd.concat([atomic_df.df['ATOM'], atomic_df.df['HETATM']], ignore_index = True)
                atomic_df.df['ATOM']['record_name'] = 'ATOM'
                atomic_df.df['HETATM'] = atomic_df.df['HETATM'].iloc[0:0]
            elif callable(append_heteroatoms):
                yn_func = append_heteroatoms
            else:
                yn_func = yes_no
            if append_heteroatoms != True:
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

        return atomic_df, filename, 'pdb'
    
    elif fileext == 'cif':
        df, cifdata = cif_to_df(filepath)
        if len(df[df['_atom_site.group_PDB']=='HETATM']) > 0:
            if append_heteroatoms == True:
                df['_atom_site.group_PDB']='ATOM'
            elif callable(append_heteroatoms):
                yn_func = append_heteroatoms
            else:
                yn_func = yes_no

            if append_heteroatoms != True:
                if yn_func('Would you like to include heteroatoms so they are used in the solvent exposure calculation?'):    
                    df['_atom_site.group_PDB']='ATOM'
                else:
                    df = df[df['_atom_site.group_PDB']=='ATOM']

        return df, filename, cifdata
    
    elif fileext == 'pkl':
        df = pd.read_pickle(filepath)
        cifpath = '_cifdata.'.join(filepath.rsplit('.', 1))
        try:
            with open(cifpath, 'rb') as f:
                cifdata = pickle.load(f)
        except Exception as e:
            print(f'Error: {e}\nCould not find cifdata')

        return df, filename, cifdata

    else:
        raise ValueError('Wrong file format; allowed file formats are .pdb, .pdb.gz, .ent, .ent.gz, .cif, .cif.gz, .pkl')


def max_m_for_full_matrix(n: int, fraction_of_avail: float = 0.6, dtype: type = np.float64) -> int:
    """
    Calculates size, m, of an m x n matrix with specified dtype that can be generated when using a specified fraction of available memory. 
    Useful for running large cdist calculations safely

    Args:
        n (int): Defined row length. e.g., total number of atoms in larger cdist entry.
        fraction_of_avail (float, optional): Maximum fraction of available memory to use for the matrix.
        dtype (type, optional): data type of matrix elements.

    Returns:
        (int): Size, n, of the largest n x n matrix to use, at maximum, the fraction of availale memory.
    """
    # fraction_of_avail: fraction of available RAM to use for the matrix
    avail = psutil.virtual_memory().available
    bytes_per_element = np.dtype(dtype).itemsize
    max_bytes = int(avail * fraction_of_avail)
    # n * m * bytes_per_element <= max_bytes  ->  m <= (max_bytes/bytes_per_element)/n
    return int(math.floor(max_bytes / bytes_per_element / n))


def preprocess(pdb_path: str,
               pre_path: str,
               yn: 'function',
               include: list = standard_atoms,
               redefine_chains: bool = False,
               pickle_out: bool = False) -> str:
    """
    Simple preprocessing of pdb and mmcif files. 
    By default, when handling pdb files, it removes all atoms that do not begin with C, N, O, or S; this normally only leaves carbon, nitrogen, oxygen, sulfur, and selenium for biological molecules.
    By default, when handling cif files, it removes all atoms that are not C, N, O, S, or Se.
    Non-standard residue names are flagged, and the user decides whether to include or not in the preprocessed file.

    Args:
        pdb_path (str): The path of the file to be preprocessed (typically pdb or mmcif).
        pre_path (str): The path of the folder inside which the preprocessed pdb will be saved.
        yn (function): For obtaining user input for yes/no questions.
        include (list, optional): If the first letter of an atom's 'atom_name' entry in the pdb/mmcif file is in this list, it will be included in the preprocessed file. If not, it will be removed.
        redefine_chains (bool, optional): If True, each chain will be relabeled, starting with A and going on alphabetically.
        pickle_out (bool, optional): If False, as default, preprocessed files will be saved as pdb/cif files.
            If True, preprocessed cif files will be saved as a pkl for the main dataframe and an accompanying pkl file for the metadata. This speeds write/read times but makes troubleshooting more difficult.

    Returns:
        out_path (str): The path of the saved preprocessed pdb.
    """
    atomic_df, filename, cifdata = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=yn)

    if cifdata == 'pdb':
        atoms = atomic_df.df["ATOM"].copy()
        atom_name, occupancy, residue_name, chain_id, residue_number = "atom_name", "occupancy", "residue_name", "chain_id", "residue_number"
    else:
        atoms = atomic_df.copy()
        atom_name, occupancy, residue_name, chain_id, residue_number = "_atom_site.label_atom_id", "_atom_site.occupancy", "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id"

    # Normalize expected columns exist
    required_cols = [atom_name, occupancy, residue_name, chain_id, residue_number]
    for c in required_cols:
        if c not in atoms.columns:
            raise KeyError(f"Expected column '{c}' in ATOM dataframe")
        
    deleted_atoms = set()
    include_h = 0
    for entry in atoms[atom_name].drop_duplicates().tolist():
        if entry not in include and entry not in deleted_atoms:
            if entry[0] == 'H':
                if include_h == 1:
                    include.append(entry)
                elif include_h == 0:
                    y = yn(f"Would you like to include hydrogen atoms?\nNot Recommended.")
                    if y:
                        include.append(entry)
                        include_h = 1
                    else:
                        include_h = 2

            else:
                include_atom = yn(f"Would you like to include atom {entry}?")
                if include_atom:
                    include.append(entry)
                else:
                    deleted_atoms.add(entry)

        
    # Base mask: atom_name is in include AND occupancy > 0.5
    # if cifdata == 'pdb':
        # atom_prefix = atoms[atom_name].astype(str).str[0]
    mask_gt_half = (atoms[atom_name].isin(include)) & (atoms[occupancy].astype(float) > 0.5)
    # elif include == ['C', 'N', 'O', 'S']:
    #     atom_prefix = atoms["_atom_site.type_symbol"].astype(str)
    #     mask_gt_half = (atom_prefix.isin(['C', 'N', 'O', 'S', 'Se', 'P'])) & (atoms[occupancy].astype(float) > 0.5)
    # else:
    #     atom_prefix = atoms[atom_name].astype(str).str[0]
    #     mask_gt_half = (atom_prefix.isin(include)) & (atoms[occupancy].astype(float) > 0.5)

    # Handle occupancy == 0.5: include only the first occurrence per (chain_id, residue_number, atom_name)
    half_mask = atoms[occupancy].astype(float) == 0.5
    if half_mask.any():
        # Build a string key safely (cast to str first)
        key = (
            atoms[chain_id].astype(str)
            + "_"
            + atoms[residue_number].astype(str)
            + "_"
            + atoms[atom_name].astype(str)
        )
        # keep the first occurrence of each key among the half-occupancy rows
        # we want: among rows where occupancy == 0.5, mark True for the first row of each key
        first_half = ~key.duplicated() & half_mask
    else:
        first_half = pd.Series(False, index=atoms.index)

    # combine masks: either >0.5 or first half-occurrence
    keep_mask = mask_gt_half | first_half

    # Non-standard residues: prompt once per residue and include/exclude all atoms of that residue
    std_res = set(standard_residues)
    residue_names = pd.Index(atoms[residue_name].astype(str).unique())
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
    residue_ok = atoms[residue_name].astype(str).isin(allowed_residues)
    residue_deleted = atoms[residue_name].astype(str).isin(deleted_residues)

    # atoms must satisfy keep_mask AND be allowed by residue selection (and not explicitly deleted)
    keep_mask &= residue_ok & (~residue_deleted)

    # 4) Optionally redefine chains:
    if redefine_chains:
        # We'll reassign chain IDs so that chain labels start at 'A' and increment when residue_number decreases
        # This replicates "relabel each chain starting with A and going on alphabetically".
        # We iterate over residues only (not per atom), building a mapping from original (chain_id,residue_number)
        # to new chain letters; then map it back to atoms.
        res_index = atoms[[chain_id, residue_number]].astype(str)
        # Compose residue-level keys maintaining original order
        res_keys = res_index[chain_id] + "_" + res_index[residue_number]
        # We want to order by the original dataframe order and detect when residue_number decreases -> new chain
        # Extract residue_numbers as integers for change detection (preserve order)
        resnums = atoms[residue_number].astype(int).to_numpy()
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
        atoms.loc[:, chain_id] = new_chain_letters

    # 5) Apply mask and save the preprocessed pdb
    if cifdata == 'pdb':
        atomic_df.df["ATOM"] = atoms.loc[keep_mask].copy()
        out_path = os.path.join(pre_path, f"{filename}.pdb")
        atomic_df.to_pdb(out_path)
    elif pickle_out:
        df = atoms.loc[keep_mask].copy()
        out_path = os.path.join(pre_path, f"{filename}.pkl")
        df.to_pickle(out_path)
        cifdata_path = os.path.join(pre_path, f"{filename}_cifdata.pkl")
        with open(cifdata_path, 'wb') as f:
            pickle.dump(cifdata, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        df = atoms.loc[keep_mask].copy()
        out_path = os.path.join(pre_path, f"{filename}.cif")
        df_to_cif(outpath=out_path, df=df, cifdata=cifdata)

    return out_path


def weights_from_df(df, yn):
    if 'atom_name' in list(df.columns.values):
        atom_name, residue_name = 'atom_name', 'residue_name'
    elif '_atom_site.label_atom_id' in list(df.columns.values):
        atom_name, residue_name = '_atom_site.label_atom_id', "_atom_site.label_comp_id"
    else:
        raise KeyError('Expected column atom_name or _atom_site.label_atom_id in df')

    weights = 12.01 * np.asarray(df[atom_name].isin(Cs)) + \
              14.01 * np.asarray(df[atom_name].isin(Ns)) + \
              16.00 * np.asarray(df[atom_name].isin(Os)) + \
              32.06 * np.asarray(df[atom_name].isin(Ss)) + \
              30.97 * np.asarray(df[atom_name].isin(Ps)) + \
              22.99 * np.asarray(df[atom_name].isin(NAs)) + \
              24.31 * np.asarray(df[atom_name].isin(MGs)) + \
              65.38 * np.asarray(df[atom_name].isin(ZNs)) + \
              55.85 * np.asarray(df[atom_name].isin(FEs)) + \
              54.94 * np.asarray(df[atom_name].isin(MNs)) + \
              35.45 * np.asarray(df[atom_name].isin(CLs)) + \
              39.10 * np.asarray(df[atom_name].isin(Ks)) + \
              58.93 * np.asarray(df[atom_name].isin(COs)) + \
              63.55 * np.asarray(df[atom_name].isin(CUs)) + \
              200.6 * np.asarray(df[atom_name].isin(HGs)) + \
              58.69 * np.asarray(df[atom_name].isin(NIs)) + \
              78.97 * np.asarray(df[atom_name].isin(SEs)) + \
              79.90 * np.asarray(df[atom_name].isin(BRs)) + \
              126.9 * np.asarray(df[atom_name].isin(Is)) + \
              87.62 * np.asarray(df[atom_name].isin(SRs)) + \
              137.3 * np.asarray(df[atom_name].isin(BAs)) + \
              28.07 * np.asarray( (df[atom_name] == 'CA') & (df[residue_name] == 'CA') ) #Differentiates calcium from alpha carbon
    
    masses = {'C' : 12.01,
              'N' : 14.01,
              'O' : 16.00,
              'S' : 32.06,
              'P' : 30.97,
              'NA': 22.99,
              'MG': 24.31,
              'ZN': 65.38,
              'FE': 55.85,
              'MN': 54.94,
              'CL': 35.45,
              'K' : 39.10,
              'CO': 58.93,
              'CU': 63.55,
              'HG': 200.6,
              'NI': 58.69,
              'SE': 78.97,
              'BR': 79.90,
              'I' : 126.9,
              'SR': 87.62,
              'BA': 137.3}
    for entry in df[atom_name].drop_duplicates().tolist():
        if entry not in all_atoms:
            if entry[0] in ['C', 'N', 'O', 'S', 'P', 'K', 'I']:
                include_atom = yn(f"Atom {entry} not recognized. Is it ok to treat this as element {entry[0]}?")
                if include_atom:
                    weights = weights + np.asarray(df[atom_name]==entry) * masses[entry[0]]
                else:
                    yn(f'Atom {entry} not recognized. Please update code.')
            elif entry[0:2] in all_atoms:
                include_atom = yn(f"Atom {entry} not recognized. Is it ok to treat this as element {entry[0:2]}?")
                if include_atom:
                    weights = weights + np.asarray(df[atom_name]==entry) * masses[entry[0:2]]
                else:
                    yn(f'Atom {entry} not recognized. Please update code.')
            else:
                yn(f'Atom {entry} not recognized. Please update code.')
    return weights


def power_cutoff(d: float|np.ndarray, constants: dict, eps: float = np.inf) -> float|np.ndarray:
    """
    Vector-safe (accepts scalars or numpy arrays for faster operation) version of scoring function. 
    - Returns 0 for distances, d, above 0
    - Returns d ** -power for distances within cutoff
    - Returns 0 for distances ≈ 0 -> atom does not count towards its own score
    - Replaces any nan/inf with finite numbers (0.0) for safety

    Args:
        d (scalar or numpy array): Distance between two atoms or array of distances between atoms.
        constant (dict): Contains the following key:value pairs:
            'power': scoring function is distance^-power
            'cutoff': at distances greater than cutoff, the scoring function returns 0.
        eps (float, optional): Epsilon used for distance ≈ 0. Purposefully infinite so that each atom does not contribute to its own score.

    Returns:
        scores (float or numpy array): Score or array of scores for input of scalar d or numpy array d, respectively.
    """

    power = constants['power']
    cutoff = constants['cutoff']

    da = np.asarray(d, dtype=float)

    # avoid warnings but still produce controlled values
    with np.errstate(divide='ignore', invalid='ignore'):
        # replace zeros with eps before inverting/squaring to avoid inf
        safe = np.where(da == 0.0, eps, da)
        scores = 1.0 / (safe ** power)

    # zero out above cutoff
    scores = np.where(da > cutoff, 0.0, scores)

    # replace any non-finite values (shouldn't be any now) with 0.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # If input was scalar, return scalar float for compatibility
    if np.isscalar(d):
        return float(scores)
    return scores
  

def power_double_cutoff(d: float|np.ndarray, constants: dict, eps: float = np.inf) -> float|np.ndarray:
    """
    Vector-safe (accepts scalars or numpy arrays for faster operation) version of scoring function. 
    - Returns 0 for distances, d, above 0
    - Returns d ** -power for distances within cutoff
    - Returns 0 for distances ≈ 0 -> atom does not count towards its own score
    - Replaces any nan/inf with finite numbers (0.0) for safety

    Args:
        d (scalar or numpy array): Distance between two atoms or array of distances between atoms.
        constant (dict): Contains the following key:value pairs:
            'power': scoring function is distance^-power
            'cutoff_far': at distances greater than cutoff_far, the scoring function returns 0.
            'cutoff_close': at distances less than cutoff_close, the scoring function returns 0.
        eps (float, optional): Epsilon used for distance ≈ 0. Purposefully infinite so that each atom does not contribute to its own score.

    Returns:
        scores (float or numpy array): Score or array of scores for input of scalar d or numpy array d, respectively.
    """

    power = constants['power']
    cutoff_far = constants['cutoff_far']
    cutoff_close = constants['cutoff_close']

    da = np.asarray(d, dtype=float)

    # avoid warnings but still produce controlled values
    with np.errstate(divide='ignore', invalid='ignore'):
        # replace zeros with eps before inverting/squaring to avoid inf
        safe = np.where(da == 0.0, eps, da)
        scores = 1.0 / (safe ** power)

    # zero out above far cutoff
    scores = np.where(da > cutoff_far, 0.0, scores)

    # zero out below close cutoff
    scores = np.where(da < cutoff_close, 0.0, scores)

    # replace any non-finite values (shouldn't be any now) with 0.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # If input was scalar, return scalar float for compatibility
    if np.isscalar(d):
        return float(scores)
    return scores


def power_close_cutoff(d: float|np.ndarray, constants: dict, eps: float = np.inf) -> float|np.ndarray:
    """
    Vector-safe (accepts scalars or numpy arrays for faster operation) version of scoring function. 
    - Returns 0 for distances, d, above 0
    - Returns d ** -power for distances outside cutoff
    - Returns 0 for distances ≈ 0 -> atom does not count towards its own score
    - Replaces any nan/inf with finite numbers (0.0) for safety

    Args:
        d (scalar or numpy array): Distance between two atoms or array of distances between atoms.
        constant (dict): Contains the following key:value pairs:
            'power': scoring function is distance^-power
            'cutoff': at distances less than cutoff, the scoring function returns 0.
        eps (float, optional): Epsilon used for distance ≈ 0. Purposefully infinite so that each atom does not contribute to its own score.

    Returns:
        scores (float or numpy array): Score or array of scores for input of scalar d or numpy array d, respectively.
    """

    power = constants['power']
    cutoff = constants['cutoff']

    da = np.asarray(d, dtype=float)

    # avoid warnings but still produce controlled values
    with np.errstate(divide='ignore', invalid='ignore'):
        # replace zeros with eps before inverting/squaring to avoid inf
        safe = np.where(da == 0.0, eps, da)
        scores = 1.0 / (safe ** power)

    # zero out above cutoff
    scores = np.where(da < cutoff, 0.0, scores)

    # replace any non-finite values (shouldn't be any now) with 0.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # If input was scalar, return scalar float for compatibility
    if np.isscalar(d):
        return float(scores)
    return scores


def exposure(pdb_path: str,
                        out_path: str,
                        yn: 'function',
                        assignment: None | dict[str, np.ndarray] = None, 
                        funcs: dict[str, 'function'] | list[dict[str, 'function']] = {'scoring_function': power_double_cutoff,
                                                                           'constants': {'power': 2, 'cutoff_far': 50, 'cutoff_close':1.85},
                                                                           'max_score': {'weight_by_amu': 423.01, 'unweighted': 31.04}},
                        weight_by_amu: bool = True,
                        progress_callback: 'function' = None):
    '''
    Saves a pdb with b-factor set to the solvent exposure score for all atoms in the pdb supplied. 
    It is recommended to run on preprocessed pdb files, so that decisions on how to handle non-standard atoms and residues are made and hydrogen atoms are removed.
    A standard scoring function and maximum score are used by default, but one in able to experiment with these if they are interested.

    Args:
        pdb_path (str): The path of the file to use in score calculation (typically pdb or mmcif). It has n atoms.
        out_path (str): The path of the folder inside which the output pdb will be saved.
        yn (function): For obtaining user input for yes/no questions.
        assignment (None or dict[str, numpy array], optional): Dictionary with value(s) that are length n numpy arrays. 
            As standard, these are boolean arrays used to obtain solvent exposure scores only accounting for the contribution of atoms with entries = True/1 while ignore the contribution of atoms with entries False/0. 
            In principle, this can be used for any algebraic operation, including calculating solvent exposure scores with weighted scores from each atom.
        funcs (dict[str: function], optional): Dictionary with value(s) calling scoring function(s) and their associated constant(s). 
        weight_by_amu (bool, optional): If True, as default, weights contributions to score by atomic mass of the paired atom. If False, doesn't.
        progress_callback (None or function, optional): If None, as default, progress messages are printed. If a function is given, custom behaviour can be implemented, such as for printing in a GUI during the run. 

    Returns:
        out (list[list[str, float, float]]): A list with one list entry (sublist) per pdb saved. 
            Each sublist contains three entries: the first is the path of the saved pdb, the second is the minimum solvent exposure score in the saved pdb, and the third is the maximum solvent exposure score in the saved pdb.

    Raises:
        TypeError: assignment must be None or dict.
    '''

    if type(funcs) == dict:
        funcs:list = [funcs]
    if type(funcs) != list:
        raise TypeError('funcs must be a dict or list of dict(s).')
    
    atomic_df, filename, cifdata = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=True)
    if cifdata == 'pdb':
        df = atomic_df.df['ATOM'].copy()
        x_coord, y_coord, z_coord, b_factor, atom_name = 'x_coord', 'y_coord', 'z_coord', 'b_factor', 'atom_name'
    else:
        df = atomic_df
        x_coord, y_coord, z_coord, b_factor, atom_name = '_atom_site.Cartn_x', '_atom_site.Cartn_y', '_atom_site.Cartn_z', '_atom_site.B_iso_or_equiv', '_atom_site.label_atom_id'
    
    coords = np.vstack((df[x_coord].to_numpy(), 
                        df[y_coord].to_numpy(), 
                        df[z_coord].to_numpy())).T
        
    out=[]

    n = len(coords)
    if weight_by_amu:
        weight_vector = weights_from_df(df=df, yn=yn)
        if assignment == None:
            assignment_vert1 = weight_vector
            assignment_vert1.shape = (n,1)
        elif type(assignment) == dict:
            assignment_vert1 = np.ones((n, len(assignment)))
            for ind, vert in enumerate(assignment.values()):
                assignment_vert1[:,ind] = np.multiply(vert, weight_vector)
            assignment_vert1.shape = (n, len(assignment))
        else:
            raise TypeError("assignment must be None or dict")
    else:
        if assignment == None:
            assignment_vert1 = np.ones((n,1), dtype=bool)
            assignment_vert1.shape = (n,1)
        elif type(assignment) == dict:
            assignment_vert1 = np.ones((n, len(assignment)), dtype=bool)
            for ind, vert in enumerate(assignment.values()):
                assignment_vert1[:,ind] = vert
            assignment_vert1.shape = (n, len(assignment))
        else:
            raise TypeError("assignment must be None or dict")
    
    assignment_vert = {}
    sums = {}
    for ind, _ in enumerate(funcs):
        assignment_vert[ind] = assignment_vert1
        sums[ind] = np.zeros_like(assignment_vert1, dtype=np.float64)

    percentages = [50]
    if n>=10000:
        percentages.append(25)
    if n>=30000:
        percentages.append(10)
    if n>=50000:
        percentages.append(75)
    if n>=100000:
        percentages.append(5)
    if n>=200000:
        percentages.append(1)
    readouts = np.array(percentages) * n // 100
    start = time.time()
    for i in range(n-1):
        d_cond = cdist([coords[i]], coords[i+1:])
        for ind, d in enumerate(funcs):
            func = d['scoring_function']
            constants = d['constants']
            if weight_by_amu:
                max_score = d['max_score']['weight_by_amu']
            else:
                max_score = d['max_score']['unweighted']
            vals = func(d_cond, constants)
            sums[ind][i:i+1] += vals @ assignment_vert[ind][i+1:]
            sums[ind][i+1:] += vals.T @ assignment_vert[ind][i:i+1]
        if i in readouts:
            current = time.time()-start
            msg = f'{round(i * 100 / n)}% complete in {current:.1f} seconds. Estimated {(n-i)*current/(i):.1f} seconds remaining.'
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    print(msg)
            else:
                print(msg)

    for indf, d in enumerate(funcs):
        func = d['scoring_function']
        constants = d['constants']
        funcname = ''
        for ke, va in constants.items():
            funcname += ke[0] + str(va).replace('.','point')
        if weight_by_amu:
            max_score = d['max_score']['weight_by_amu']
        else:
            max_score = d['max_score']['unweighted']
        if assignment:
            ind=0
            for ind, k in enumerate(assignment.keys()):
                if max_score == 0:
                    df[b_factor] = sums[indf][:,ind]
                else:
                    df[b_factor] = 100 / max_score * (max_score - sums[indf][:,ind])

                if cifdata == 'pdb':
                    outname = os.path.join(out_path, filename + '_' + k + '_' + funcname + '.pdb')
                    atomic_df.df['ATOM'] = df.copy()
                    atomic_df.to_pdb(outname)
                else:
                    outname = os.path.join(out_path, filename + '_' + k + '_' + funcname + '.cif')
                    df_to_cif(outname, df=df, cifdata=cifdata)
                out += [[outname, min(df[b_factor]), max(df[b_factor])]]
                ind+=1
        else:
            if max_score == 0:
                df[b_factor] = sums[indf]
            else:
                df[b_factor] = 100 / max_score * (max_score - sums[indf])
            if cifdata == 'pdb':
                outname = os.path.join(out_path, filename + '_' + funcname + '.pdb')
                atomic_df.df['ATOM'] = df.copy()
                atomic_df.to_pdb(outname)
            else:
                outname = os.path.join(out_path, filename + '_' + funcname + '.cif')
                df_to_cif(outname, df=df, cifdata=cifdata)
            out += [[outname, min(df[b_factor]), max(df[b_factor])]]
    
    return out


def max_exposure_score(funcs: dict[str, 'function'] | list[dict[str, 'function']],
                       assignment: None | dict[str, np.ndarray],
                       subsample: bool | int,
                       yn: 'function',
                       weight_by_amu: bool = True,
                       n_sigmas: float | int = 2) -> pd.DataFrame:
    '''
    Calculates the maximum score for input functions and assignment vectors using RuBisCO with water modelled around.

    Args:
        funcs (dict[str: function], optional): Dictionary with value(s) calling scoring function(s) and their associated constant(s). 
        assignment (None or dict[str, numpy array], optional): Dictionary with value(s) that are length n numpy arrays. 
            As standard, these are boolean arrays used to obtain solvent exposure scores only accounting for the contribution of atoms with entries = True/1 while ignore the contribution of atoms with entries False/0. 
            In principle, this can be used for any algebraic operation, including calculating solvent exposure scores with weighted scores from each atom.
        subsample (bool or int): Number of atoms to subsample from RuBisCO asymmetric subunit to yield scores. If True, 1600. If False, all atoms. Warning, may require more memory than available if False.
        yn (function): For obtaining user input for yes/no questions.
        weight_by_amu (bool, optional): If True, as default, weights contributions to score by atomic mass of the paired atom. If False, doesn't.
        n_sigmas (float or int, optional): Maximum score = mean(scores) + n_sigmas * std(scores)


    '''
    if type(funcs) == dict:
        funcs:list = [funcs]
    if type(funcs) != list:
        raise TypeError('funcs must be a dict or list of dict(s).')
    
    df = pd.read_pickle(os.path.join(basedir, 'standards', 'rubisco.pkl'))
    coords_all = np.vstack((df['_atom_site.Cartn_x'].to_numpy(), 
                            df['_atom_site.Cartn_y'].to_numpy(), 
                            df['_atom_site.Cartn_z'].to_numpy())).T
    prot_df = df[df['_atom_site.label_asym_id'] == 'A']
    coords_prot = np.vstack((prot_df['_atom_site.Cartn_x'].to_numpy(), 
                             prot_df['_atom_site.Cartn_y'].to_numpy(), 
                             prot_df['_atom_site.Cartn_z'].to_numpy())).T

    if assignment == None:
        assignment = {'tot': np.ones(len(coords_all))}

    if weight_by_amu:
        weight_vector = weights_from_df(df=df, yn=yn)

    # Make do just one subunit
    coords_prot_sampled = coords_prot[:int(len(coords_prot)/8)]

    # Include n atoms -> faster
    if subsample != False:
        if subsample == True:
            subsample = 1600
        sampled_indices = np.linspace(0, len(coords_prot_sampled)-1, subsample, dtype=np.uint32)
        coords_prot_sampled = coords_prot_sampled[sampled_indices]
        
    df_out = pd.DataFrame(np.zeros((len(funcs), len(assignment))), index=list(assignment.keys()))
    
    scores = {}
    for k,v in assignment.items():
        scores[k] = np.zeros((len(funcs), len(coords_prot_sampled)))

    for ind, coord in enumerate(coords_prot_sampled):
        d_cond = cdist([coord], coords_all)
        for i, d in enumerate(funcs):
            func = d['scoring_function']
            constants = d['constants']
            vals = func(d_cond, constants)
            for k,v in assignment.items():
                if weight_by_amu:
                    v = np.multiply(v, weight_vector)
                scores[k][i, ind:ind+1] = np.dot(vals, v)

    for i, d in enumerate(funcs):
        for k,v in assignment.items():
            df_out.loc[k,i] = np.mean(scores[k][i]) + n_sigmas * np.std(scores[k][i])
    
    return df_out


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
        atomic_df, filename, cifdata = read_pdb_mmcif(filepath=filepath)

        if cifdata == 'pdb':
            df = atomic_df.df['ATOM'].copy()
            b_factor = 'b_factor'
            atom_name, chain_id, residue_number = "atom_name", "chain_id", "residue_number"
        else:
            df = atomic_df
            b_factor = '_atom_site.B_iso_or_equiv'
            atom_name, chain_id, residue_number = "_atom_site.label_atom_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id"

        current_residue = df.iloc[0].loc[chain_id] + str(df.iloc[0].loc[residue_number])

        scores = np.zeros(len(df))
        backbone_scores = np.zeros(len(df))

        atom_count = 0
        score = 0
        backbone_atom_count = 0
        backbone_score = 0

        for i, x in df.iterrows():
            if x[chain_id] + str(x[residue_number]) == current_residue:
                atom_count+=1
                score+=x[b_factor]
                if x[atom_name] in ['C', 'N', 'O', 'CA']:
                    backbone_atom_count+=1
                    backbone_score+=x[b_factor]
                if x.equals(df.iloc[-1]):
                    for j in range(atom_count):
                        scores[i-j] = score / atom_count
                        if backbone_atom_count != 0:
                            backbone_scores[i-j] = backbone_score / backbone_atom_count

                    atom_count=1
                    score=x[b_factor]

            else:
                for j in range(atom_count):
                    scores[i-j-1] = score / atom_count
                    if backbone_atom_count != 0:
                        backbone_scores[i-j-1] = backbone_score / backbone_atom_count

                atom_count=1
                score=x[b_factor]
                current_residue = x[chain_id] + str(x[residue_number])

                if x[atom_name] in ['C', 'N', 'O', 'CA']:
                    backbone_atom_count=1
                    backbone_score=x[b_factor]
                else:
                    backbone_atom_count=0
                    backbone_score=0

                if i == len(df) - 1:
                    scores[i] = score
                    backbone_scores[i] = score

        df[b_factor] = scores
        if cifdata == 'pdb':
            outname = filepath.split('.',1)[0] + '_avgbyres.pdb'
            atomic_df.df['ATOM'] = df.copy()
            atomic_df.to_pdb(outname)
        else:
            outname = filepath.split('.',1)[0] + '_avgbyres.cif'
            df_to_cif(outname, df=df, cifdata=cifdata)
        out += [[outname, min(scores), max(scores)]]

        if backbone:
            if cifdata == 'pdb':
                outname = filepath.split('.',1)[0] + '_avgbyresbb.pdb'
                atomic_df.df['ATOM'] = df.copy()
                atomic_df.to_pdb(outname)
            else:
                outname = filepath.split('.',1)[0] + '_avgbyresbb.cif'
                df_to_cif(outname, df=df, cifdata=cifdata)
            out += [[outname, min(backbone_scores), max(backbone_scores)]]

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
    if pdb_path.rsplit('.',1)[1] == 'pkl':
        df = pd.read_pickle(pdb_path)
    else: 
        atomic_df, _, cifdata = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=True)

        if cifdata == 'pdb':
            df = atomic_df.df['ATOM'].copy()
        else:
            df = atomic_df

    out_tot = np.ones(len(df), dtype=bool)

    if type(chain1) == str:
        out1 = np.array(df[feature].eq(chain1)).astype(bool)
        name = chain1
    elif type(chain1) == list:
        for ind, chain in enumerate(chain1):
            if ind == 0:
                temp = df[feature].eq(chain)
                name = chain
            else:
                name = name + '_' + chain
                temp = temp | df[feature].eq(chain)
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
    if pdb_path.rsplit('.',1)[1] == 'pkl':
        df = pd.read_pickle(pdb_path)
    else:
        atomic_df, _, cifdata = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=True)

        if cifdata == 'pdb':
            df = atomic_df.df['ATOM'].copy()
        else:
            df = atomic_df


    if type(include) == str:
        return {include: np.array(df[feature].eq(include)).astype(bool)}
    elif type(include) == list:
        for ind, chain in enumerate(include):
            if ind == 0:
                temp = df[feature].eq(chain)
                name = chain
            else:
                name = name + '_' + chain
                temp = temp | df[feature].eq(chain)
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

        atomic_df, _, cifdata = read_pdb_mmcif(filepath=pdb_path)

        if cifdata == 'pdb':
            df = atomic_df.df['ATOM'].copy()
            x_coord, y_coord, z_coord, b_factor = 'x_coord', 'y_coord', 'z_coord', 'b_factor'
            atom_name, occupancy, residue_name, chain_id, residue_number = "atom_name", "occupancy", "residue_name", "chain_id", "residue_number"
        else:
            df = atomic_df
            x_coord, y_coord, z_coord, b_factor = '_atom_site.Cartn_x', '_atom_site.Cartn_y', '_atom_site.Cartn_z', '_atom_site.B_iso_or_equiv'
            atom_name, occupancy, residue_name, chain_id, residue_number = "_atom_site.label_atom_id", "_atom_site.occupancy", "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id"


        i=0

        errorcount = 0

        if localres.iloc[0].name[0] == '#':
            k = len( localres.iloc[0].name.split('/')[0] )
        else:
            k = 0

        for ind, row in localres.iterrows():
            if not backboneonly or ind.split('@')[1] in ['C', 'N', 'O', 'CA']:
                try:
                    if type(df.loc[(ind[1+k:2+k], int(ind[3+k:].split('@')[0])), b_factor][ind[3+k:].split('@')[1]]) == pd.core.series.Series:
                        errorcount+=1
                    else:
                        out[i,0] = df.loc[(ind[1+k:2+k], int(ind[3+k:].split('@')[0])), b_factor][ind[3+k:].split('@')[1]]
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


def score_v_localres_plotly(pdb_path: str, 
                     defattr_path: str, 
                     only_chain: bool | list[str] = False,
                     backboneonly: bool = False, 
                     append_heteroatoms: 'function' = yes_no):
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
    localres = pd.read_csv(defattr_path, sep = '\t', header = 3, usecols = [1,2], names = ['atom', 'localres']).set_index('atom')
    localres['localres'] = 1.0/localres['localres']

    # out = np.zeros((len(localres), 2))
    # names = list(np.zeros(len(localres)).astype(int).astype(str))

    atomic_df, _, cifdata = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=append_heteroatoms)

    if cifdata == 'pdb':
        df = atomic_df.df['ATOM'].copy()
        x_coord, y_coord, z_coord, b_factor = 'x_coord', 'y_coord', 'z_coord', 'b_factor'
        atom_name, chain_id, residue_number = "atom_name", "chain_id", "residue_number"
    else:
        df = atomic_df
        x_coord, y_coord, z_coord, b_factor = '_atom_site.Cartn_x', '_atom_site.Cartn_y', '_atom_site.Cartn_z', '_atom_site.B_iso_or_equiv'
        atom_name, chain_id, residue_number = "_atom_site.label_atom_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id"

    df['index'] = '/'+df[chain_id]+':'+df[residue_number].astype(str)+'@'+df[atom_name]

    if cifdata == 'pdb':
        data = pd.concat([df.set_index('index'), localres], axis=1).drop(['charge', 'blank_1', 'alt_loc', 'blank_2', 'blank_3', 'insertion', 'blank_4', 'segment_id', 'element_symbol'], axis=1).dropna()
    else:
        data = pd.concat([df.set_index('index'), localres], axis=1).dropna('all', axis=1).dropna()

    if only_chain:
        data = data[(data[chain_id].isin(only_chain))]

    if backboneonly:
        data = data[(data[atom_name].isin(['C', 'N', 'O', 'CA']))]

    fig = px.scatter(data, 
                     x=b_factor, y='localres',
                     hover_data={b_factor:False, 
                                 'localres':False, 
                                 'Atom':data.index,
                                 'Score':data[b_factor],
                                 'Local Res':data['localres']}
                     )

    fig.update_traces(marker_line_width=0, marker=dict(opacity=1.0))
    
    fig.update_xaxes(title_text='Solvent Exposure Score (Arbitrary Units)')
    
    fig.update_yaxes(title_text='Local Resolution at Atom in Model (1/Å)')

    fig.update_layout(
        yaxis = dict(
            tickmode = 'array',
            tickvals = [0.5, 0.33, 0.25, 0.2, 0.166666],
            ticktext = ['1/2', '1/3', '1/4', '1/5', '1/6']
        )
    )

    return fig


def visualize(pdb_path: str,
              b_factor_range: list = [0, 100],
              append_heteroatoms: 'function' = yes_no) -> 'plotly.graph_objs._figure.Figure':
    """
    Builds and returns a Plotly Figure for the given pdb_path.

    Args:
        pdb_path (str): The path of the pdb file to be visualized.
        b_factor_range (list, optional): The min and max value of the colorscale. 
            Default 0 to 20, as standard for solvent exposure scores using score = d**-2 with cutoff of 5 nm.
        append_heteroatoms (function, optional): If there are heteroatoms in the read file, the output of this function decides if any heteroatoms are added to the chain atoms for later use. If False, any are removed.

    Returns:
        fig (Figure): Plotly figure of atoms.
    """
    atomic_df, _, cifdata = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=append_heteroatoms)

    if cifdata == 'pdb':
        df = atomic_df.df['ATOM'].copy()
        x_coord, y_coord, z_coord, b_factor = 'x_coord', 'y_coord', 'z_coord', 'b_factor'
        atom_name, chain_id, residue_number = "atom_name", "chain_id", "residue_number"
    else:
        df = atomic_df
        x_coord, y_coord, z_coord, b_factor = '_atom_site.Cartn_x', '_atom_site.Cartn_y', '_atom_site.Cartn_z', '_atom_site.B_iso_or_equiv'
        atom_name, chain_id, residue_number = "_atom_site.label_atom_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id"



    backbone_atoms = ['C', 'N', 'O', 'CA']
    df['marker_size'] = df[atom_name].apply(
        lambda x: 4 if x in backbone_atoms else 2
    )

    fig = px.scatter_3d(
        df,
        x=x_coord, y=y_coord, z=z_coord,
        hover_data={x_coord:False, 
                    y_coord:False, 
                    z_coord:False, 
                    'chain':df[chain_id],
                    'Residue':df[residue_number],
                    'Atom':df[atom_name],
                    'marker_size': False,
                    'Score':df[b_factor],
                    b_factor:False},
        color=b_factor,
        color_continuous_scale=[
            (0, '#ffffff'),
            (0.25, '#ffff00'),
            (0.5, '#ff0000'),
            (0.75, '#000088'),
            (1, '#000000')
        ],
        range_color=b_factor_range,
        size='marker_size'
    )

    fig.update_layout(coloraxis_colorbar=dict(title="Solvent Exposure Score"))

    fig.update_traces(marker_line_width=0, marker=dict(opacity=1.0))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                title='',
                visible=False,
                backgroundcolor='rgba(0,0,0,0)'
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                title='',
                visible=False,
                backgroundcolor='rgba(0,0,0,0)'
            ),
            zaxis=dict(
                showgrid=False,
                showticklabels=False,
                title='',
                visible=False,
                backgroundcolor='rgba(0,0,0,0)'
            ),
            bgcolor='rgba(0,0,0,0)'  # removes scene background
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # removes outer background
        plot_bgcolor='rgba(0,0,0,0)',   # removes plot background
    )

    return fig


def getcols(pdb_path: str, yn: 'function' = yes_no) -> list:
    atomic_df, _, cifdata = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=yn)
    if cifdata == 'pdb':
        df = atomic_df.df['ATOM']
    else:
        df = atomic_df

    df = df.dropna(axis=1)

    return list(df.columns.values)


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
    atomic_df, _, cifdata = read_pdb_mmcif(filepath=pdb_path, append_heteroatoms=yn)
    if cifdata == 'pdb':
        return atomic_df.df['ATOM'][feature].drop_duplicates().tolist()
    else:
        return atomic_df[feature].drop_duplicates().tolist()


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


available_scoring_functions = {'Far cutoff': {'scoring_function': power_cutoff,
                                         'constants': {'power': 2, 'cutoff': 50},
                                         'max_score': {'weight_by_amu': 436.89, 'unweighted': 32.08}},
                               'Close and far cutoff': {'scoring_function': power_double_cutoff,
                                          'constants': {'power': 3, 'cutoff_far': 50, 'cutoff_close': 1.85},
                                          'max_score': {'weight_by_amu': 30, 'unweighted': 2.22}},
                               'Close cutoff': {'scoring_function': power_close_cutoff,
                                          'constants': {'power': 3.5, 'cutoff': 1.85},
                                          'max_score': {'weight_by_amu': 11.82, 'unweighted': 0.87}},
    }


