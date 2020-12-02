
def as_std_type(value):
    """Convert numpy type to standard type."""
    return getattr(value, "tolist", lambda: value)()

def make_seg_id_seg_pdb_dict(glob_pattern='*_[0-9][0-9][0-9].pdb'):
    """
    Simply globs all files matching a certain pattern and assigns a 4-letter ID to each of them.
    Args:
        glob_pattern (str)
    """
    from glob import glob

    def four_letter_id_from_int(n):
        return chr( (n // 26**3) % 26 + ord('A') ) \
             + chr( (n // 26**2) % 26 + ord('A') ) \
             + chr( (n // 26**1) % 26 + ord('A') ) \
             + chr( (n // 26**0) % 26 + ord('A') )

    def four_letter_id(max=456976): # 26**4
        """generator for consectutive 4 letter ids"""
        for n in range(0,max):
            yield four_letter_id_from_int(n)

    alpha_id = four_letter_id()

    seg_id_seg_pdb_dict = {
        next(alpha_id): pdb for pdb in sorted( glob(glob_pattern) ) }

    print("Found these segments: {}".format(seg_id_seg_pdb_dict))

    return seg_id_seg_pdb_dict

    # FWAction(
    #     stored_data = { 'seg_id_seg_pdb_dict' : seg_id_seg_pdb_dict},
    #     mod_spec = [ { '_set' : { 'context->segments' : seg_id_seg_pdb_dict } } ] )