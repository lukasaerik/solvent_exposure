import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
import matplotlib.pyplot as plt
import os, math, time

basedir = os.path.dirname(__file__)
standard_residues = ['LYS', 'LEU', 'THR', 'TYR', 'PRO', 'GLU', 'ASP', 'ILE', 'ALA', 'PHE', 'ARG',
                     'VAL', 'GLN', 'GLY', 'SER', 'TRP', 'CYS', 'HIS', 'ASN', 'MET', 'SEC', 'PYL']


def yes_no(text):
    yn = input(text + 'y/n')
    if yn == 'y':
        return True
    else:
        return False


def preprocess(pdb_path, 
               pre_path,
               yes_no = yes_no,
               feature: str = 'atom_name', 
               include: list = ['C', 'N', 'O', 'S'], 
               redefine_chains: bool = False):
    
    try:
        filename = pdb_path.rsplit('/',1)[1]
    except IndexError:
        try:
            filename = pdb_path.rsplit("\\",1)[1]
        except IndexError:
            filename = pdb_path


    atomic_df = PandasPdb().read_pdb(pdb_path)

    atomic_df = atomic_df.get_model(1)

    atomic_df.df['ATOM'] = atomic_df.df['ATOM'].drop('model_id', axis = 1)

    atomic_df.df['ATOM']['segment_id'] = ''
    atomic_df.df['ATOM']['element_symbol'] = ''

    temp = atomic_df.df['ATOM'][feature].eq(include[0])

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
            if x[feature][0] in include and x['occupancy'] > 0.5:
            
                temp.loc[i] = True

                if x['residue_number'] >= residue_counter and redefine_chains:
                    atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                    residue_counter = x['residue_number']
                elif redefine_chains:
                    chain += 1
                    atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                    residue_counter = x['residue_number']

            if x[feature][0] in include and x['occupancy'] == 0.5:
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
            yn = yes_no(f"Would you like to include residue {x['residue_name']}?")
            if yn:
                added_residues += [x['residue_name']]
                if x[feature][0] in include and x['occupancy'] > 0.5:
                
                    temp.loc[i] = True

                    if x['residue_number'] >= residue_counter and redefine_chains:
                        atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                        residue_counter = x['residue_number']
                    elif redefine_chains:
                        chain += 1
                        atomic_df.df['ATOM'].loc[i, 'chain_id'] = chr(chain)
                        residue_counter = x['residue_number']

                if x[feature][0] in include and x['occupancy'] == 0.5:
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
    out_path = os.path.join(pre_path, filename)
    atomic_df.to_pdb(out_path)
    return out_path


def f2_cutoff(x, cutoff: float = 50):
    if x > cutoff:
        return 0
    else:
        return x ** -2
    

def exposure(pdb_path,
             out_path,
             funcs: dict = {'2c50': f2_cutoff}, 
             assignment=None, 
             max_scores: dict = {'2c50': 26.5}, 
             save_matrix: bool = False, 
             save_scores_as_vector: bool = False,
             progress_callback = None):

    try:
        filename = pdb_path.rsplit('/',1)[1]
    except IndexError:
        try:
            filename = pdb_path.rsplit("\\",1)[1]
        except IndexError:
            filename = pdb_path
            
    filename, fileext = filename.rsplit('.',1)
    fileext = '.' + fileext 

    atomic_df = PandasPdb().read_pdb(pdb_path)

    atomic_df = atomic_df.get_model(1)

    atomic_df.df['ATOM'] = atomic_df.df['ATOM'].drop('model_id', axis = 1)


    coords = np.vstack((atomic_df.df['ATOM']['x_coord'].to_numpy(), 
                        atomic_df.df['ATOM']['y_coord'].to_numpy(), 
                        atomic_df.df['ATOM']['z_coord'].to_numpy())).T

    pair_scores = {}

    out = []

    n = len(coords)
    for key in funcs:
        pair_scores[key] = np.zeros((n, n))

    print_percentages = [1,5,25,50,75]
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
            atomic_df.to_pdb(os.path.join(out_path, filename + '_' + key + fileext))

            out += [[os.path.join(out_path, filename + '_' + key + fileext), min(max_scores[key] - temp), max(max_scores[key] - temp)]]
            
            if save_scores_as_vector:
                np.save(os.path.join(out_path, filename + '_' + key + '_vec.npy', temp))

            if save_matrix:
                np.save(os.path.join(out_path, filename + '_' + key + '_mat.npy', mat))
    
    if type(assignment) == dict:
        mat_not_saved = True
        for k, assigment_vert in assignment.items():
            for key, mat in pair_scores.items():
                temp = assigment_vert @ mat
                atomic_df.df['ATOM']['b_factor'] = max_scores[key] - temp
                atomic_df.to_pdb(os.path.join(out_path, filename + '_' + k + '_' + key + fileext))
                out += [[os.path.join(out_path, filename + '_' + k + '_' + key + fileext), min(max_scores[key] - temp), max(max_scores[key] - temp)]]

                if save_scores_as_vector:
                    np.save(os.path.join(out_path, filename + '_' + k + '_' + key + '_vec.npy'), max_scores[key] - temp)

                if save_matrix and mat_not_saved:
                    np.save(os.path.join(out_path, filename + '_' + key + '_mat.npy'), mat)
                    mat_not_saved = False

    return out


def average_score(pdb_path):
    if pdb_path[-3:] == 'pdb':
        atomic_df = PandasPdb().read_pdb(pdb_path)

        atomic_df = atomic_df.get_model(1)

        atomic_df.df['ATOM'] = atomic_df.df['ATOM'].drop('model_id', axis = 1)

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

        atomic_df.df['ATOM']['b_factor'] = scores
        atomic_df.to_pdb(pdb_path[:-4] + '_avgbyres.pdb')

        atomic_df.df['ATOM']['b_factor'] = backbone_scores
        atomic_df.to_pdb(pdb_path[:-4] + '_avgbyresbb.pdb')

    elif pdb_path[-7:] == 'defattr':
        localres = pd.read_csv(pdb_path, sep = '\t', header = 3, usecols = [1,2], names = ['atom', 'localres']).set_index('atom')

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

        score_df = pd.read_csv(pdb_path, sep = '\t', header = 3, names = ['', 'atom', 'localres'])
        score_df['localres'] = scores
        score_df.to_csv(pdb_path[:-8] + '_avgbyres.defattr', sep = '\t', header=['attribute: locres \nrecipient: atoms \nmatch mode: 1-to-1', '', ''], index=False)

        score_df['localres'] = backbone_scores
        score_df.to_csv(pdb_path[:-8] + '_avgbyresbb.defattr', sep = '\t', header=['attribute: locres \nrecipient: atoms \nmatch mode: 1-to-1', '', ''], index=False)


def create_3_vectors(pdb_path, chain1, feature):
    atomic_df = PandasPdb().read_pdb(pdb_path)
    atomic_df = atomic_df.get_model(1)
    atomic_df.df['ATOM'] = atomic_df.df['ATOM'].drop('model_id', axis = 1)

    out_tot = np.ones(len(atomic_df.df['ATOM']))

    if type(chain1) == str:
        out1 = np.array(atomic_df.df['ATOM'][feature].eq(chain1)).astype(int)
        name = chain1
    elif type(chain1) == list:
        for ind, chain in enumerate(chain1):
            if ind == 0:
                temp = atomic_df.df['ATOM'][feature].eq(chain)
                if len(chain1) == 1:
                    name = chain
                else:
                    name = chain + 'plus'
            else:
                temp = temp | atomic_df.df['ATOM'][feature].eq(chain)
        out1 = np.array(temp).astype(int)

    out2 = out_tot-out1


    return {name: out1, 'not'+name: out2, 'tot': out_tot}


def score_v_localres(pdb_path, 
                     defattr_path, 
                     only_chain = False,
                     called_by_GUI:bool = False, 
                     backboneonly: bool = False, 
                     inverse: bool = True, 
                     interactive: bool = False):
    if called_by_GUI:
        localres = pd.read_csv(defattr_path, sep = '\t', header = 3, usecols = [1,2], names = ['atom', 'localres']).set_index('atom')

        out = np.zeros((len(localres), 2))
        names = list(np.zeros(len(localres)).astype(int).astype(str))

        atomic_df = PandasPdb().read_pdb(pdb_path)
        atomic_df = atomic_df.get_model(1)
        atomic_df.df['ATOM'] = atomic_df.df['ATOM'].drop('model_id', axis = 1)
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

        atomic_df = PandasPdb().read_pdb(pdb_path)

        atomic_df = atomic_df.get_model(1)

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

            try:
                # For ipympl in notebooks, explicit show helps
                plt.show()
            except Exception:
                plt.ion()
                plt.show()
            return {'fig': fig, 'ax': ax, 'sc': sc, 'names': names}
        else:
            plt.show()
            return {'fig': fig, 'ax': ax, 'sc': sc, 'names': names}


def features(pdb_path, feature):
    atomic_df = PandasPdb().read_pdb(pdb_path)
    atomic_df = atomic_df.get_model(1)
    atomic_df.df['ATOM'] = atomic_df.df['ATOM'].drop('model_id', axis = 1)

    out = []

    for chain in atomic_df.df['ATOM'][feature]:
        if chain not in out:
            out = out + [chain]

    return out


def reciprocal_ticks(mn, mx, n = 4, intervals = [1,2,5,10]):
    ticks = []
    for i in intervals:
        if i/mn - i/mx < n:
            None
        else:
            tick = math.ceil(i/mn)/i
            while tick > 1/mx:
                ticks += [tick]
                tick += -1/i
            break
    return np.array(ticks)
    