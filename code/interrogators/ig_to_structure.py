# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:59:16 2020
@author: Alex Vinogradov
"""

import os
from rdkit.Chem import AllChem  
from rdkit import Chem
import utils.Plotter as Plotter
from utils import constants, feature_matrix_generators

import matplotlib.colors

def _seq_to_rdk(PEP):
    #convert an ndarray (PEP) to a list of RDkit molecule objects
    return [Chem.MolFromSmiles(constants.aaSMILES[constants.aas.index(x)]) for x in PEP]

def _convert_attributions_to_structure(PEP, IG, r):
    '''
    for each amino acid in the peptide,
        for each atom and each bond in amino acid
            compute the total IG attribution sum.
    
    If an atom or a bond constitutes a part of a fingerprint,
    it receives the full score for the fingeprint (i.e. normalization
    to, say, the total number of atoms in the fingerprint is not done).
    
    PEP : 1D ndarray with amino acids as letter (dtype ~ <U1)
    IG : integrated gradients for peptide PEP; has to be computed elsewhere
    r  : int; atomic radius for the fingerprint generator; has to be the same
         as the value used during featurization and training.

    out : full_pep_atom_attrs, a list of dicts containing attribution values
          for each atom in each amino acids
          full_pep_bond_attrs same, but for bonds
    '''
    
    F, bits, info = feature_matrix_generators.dense_morgan(r, w=False)
    full_pep_atom_attrs = list()
    full_pep_bond_attrs = list()    
    
    rdk_peptide = _seq_to_rdk(PEP)
        
    for pos, aa in enumerate(PEP):
    
        m = rdk_peptide[pos]
        f = AllChem.GetMorganFingerprint(m, 4).GetNonzeroElements()
        attributions_at_aa = IG[pos]    
    
        #initiate a dict of attributions for every atom and bond in  aa
        aa_atom_attributions = dict()
        for atom in m.GetAtoms():
            aa_atom_attributions[atom.GetIdx()] = 0
        
        aa_bond_attributions = dict()
        for bond in m.GetBonds():
            aa_bond_attributions[bond.GetIdx()] = 0
    
        for key in f.keys():
            
            fp_attribution = attributions_at_aa[bits.index(key)]
    
            #a single FP may contain several submolecules apparently
            for sub in info[constants.aas.index(aa)][key]:
                atom_id, radius = sub
       
                env = Chem.FindAtomEnvironmentOfRadiusN(m, radius, atom_id)
                atoms = set([atom_id])
                bonds = set()
                            
                for bidx in env:
                    atoms.add(m.GetBondWithIdx(bidx).GetBeginAtomIdx())
                    atoms.add(m.GetBondWithIdx(bidx).GetEndAtomIdx())
                    bonds.add(bidx)            
    
            for a in atoms:
                aa_atom_attributions[a] += fp_attribution
            for b in bonds:
                aa_bond_attributions[b] += fp_attribution
            
        full_pep_atom_attrs.append(aa_atom_attributions)        
        full_pep_bond_attrs.append(aa_bond_attributions)        
        
    return full_pep_atom_attrs, full_pep_bond_attrs


def _normalize_attribution_colors(full_pep_atom_attrs, full_pep_bond_attrs):
    
    combined = full_pep_atom_attrs + full_pep_bond_attrs
    
    x = [list(x.values() ) for x in combined]
    x = [item for sublist in x for item in sublist]
    
    return matplotlib.colors.Normalize(vmin=min(x), vmax=max(x))


def _convert_attributions_to_rgb(attrs, cmap, norm):

    for aa_atr in attrs:
        for elem in aa_atr:
            aa_atr[elem] = cmap(norm(aa_atr[elem]))[:-1]        

    return attrs

def attributions_to_structure(PEP, IG, cmap, r):
    '''
    Parameters
    ----------
    PEP : 1D ndarray with amino acids as letter (dtype ~ <U1)
    IG : integrated gradients for peptide PEP; has to be computed elsewhere
    cmap : matplotlib-compatible colormap instance

    Returns
    -------
    None; two .svg (one with attributions mapped to structure, and another one is a colorbar)
          will be created.

    '''
    
    atom_wise_attrs, bond_wise_attrs = _convert_attributions_to_structure(PEP, IG, r)    
   
    norm = _normalize_attribution_colors(atom_wise_attrs, bond_wise_attrs)       
    atom_rgbs = _convert_attributions_to_rgb(atom_wise_attrs, cmap, norm)
    bond_rgbs = _convert_attributions_to_rgb(bond_wise_attrs, cmap,norm) 
    
    atom_list = [list(d.keys()) for d in atom_rgbs] 
    bond_list = [list(d.keys()) for d in bond_rgbs] 
    rdk_pep = _seq_to_rdk(PEP)
    
    #align representation along the common (Gly) core    
    Gly = Chem.MolFromSmiles(constants.aaSMILES[constants.aas.index('G')])
    subms = [x for x in rdk_pep if x.HasSubstructMatch(Gly)]
    AllChem.Compute2DCoords(Gly)
    
    for m in subms:
        _ = AllChem.GenerateDepictionMatching2DStructure(m, Gly)
        
    img = Chem.Draw.MolsToGridImage(subms, 
                                    molsPerRow=11, 
                                    highlightAtomLists=atom_list, 
                                    highlightAtomColors=atom_rgbs,
                                    highlightBondLists=bond_list,
                                    highlightBondColors=bond_rgbs,
                                    useSVG=True
                                   )    
    
    if not os.path.isdir('../peptide_interrogation/attributions_to_structure'):
        os.makedirs('../peptide_interrogation/attributions_to_structure')
    
    s = ''.join(PEP)
    str_path = os.path.join('../peptide_interrogation/attributions_to_structure', f'peptide_{s}_strutural_attributions.svg')
    with open(str_path, 'w') as f:
        f.write(img.data)         
    
    cb_path = os.path.join('../peptide_interrogation/attributions_to_structure', f'peptide_{s}_colorbar.svg')
    Plotter.attribution_colorbar(norm, cmap, cb_path)
    
    return
    






        
            