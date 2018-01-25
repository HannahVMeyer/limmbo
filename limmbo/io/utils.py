from os.path import exists

def file_type(filepath):
    r"""
    Determine filetype based on file ending
    Arguments:
        filepath (string):
            'path/2/file/with/fileending
    Returns:
        (string):
            filetype

    Example:

    .. doctest::
        
    """
    imexts = ['.png', '.bmp', '.jpg', 'jpeg']
    textexts = ['.csv', '.txt']
    if filepath.endswith('.hdf5') or filepath.endswith('.h5'):
        return 'hdf5'
    if any([filepath.endswith(ext) for ext in textexts]):
        return 'delim'
    if filepath.endswith('.grm.raw'):
        return 'grm.raw'
    if filepath.endswith('.npy'):
        return 'npy'
    if _is_bed(filepath):
        return 'bed'
    if _is_gen(filepath):
        return 'gen'
    if any([filepath.endswith(ext) for ext in imexts]):
        return 'image'
    return 'unknown'


def _is_bed(filepath):
    return all([exists(filepath + ext) for ext in ['.bed', '.bim', '.fam']])

def _is_gen(filepath):
    return all([exists(filepath + ext) for ext in ['.gen', '.sample']])
