

lmp_type_ase_element_mapping = {
    '11': 'Au',
}

ase_type_pmd_type_mapping = {
    'Au': 'AU',
}

ase_type_pmd_residue_mapping = {
    'Au': 'AUM',
}

pdb_residue_charmm_residue_mapping = {
    'SOL': 'TIP3',
    'NA': 'SOD',
    'AUM': 'AUM',
    'SDS': 'SDS',
}


pdb_type_charmm_type_mapping = {
    'TIP3': {
        'OW': 'OH2',
        'HW1': 'H1',
        'HW2': 'H2',
    },
    'SOD': {
        'NA': 'SOD',
    },
    'AUM': {
        'AU': 'AU'
    },
    'SDS': {}
    # SDS names don't change
}

psfgen_mappings_template_context = {
    'residues': [
        {
            'in': res_in,
            'out': res_out,
            'atoms': [
                {
                    'in': atm_in,
                    'out': atm_out,
                } for atm_in, atm_out in pdb_type_charmm_type_mapping[res_out].items()
            ]
        } for res_in, res_out in pdb_residue_charmm_residue_mapping.items()
    ]
}