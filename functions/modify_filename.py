import os
from pathlib import Path , PureWindowsPath
def modify_filename(PATH, filename):
    """
    Modify a filename to make it unique if it already exists in a specified directory.

    Parameters:
        PATH (str): The path to the directory where the file is located or will be saved.
        filename (str): The original filename.

    Returns:
        str: A modified filename that is unique in the specified directory. If the
             original filename does not exist in the directory, it is returned unchanged.
    """
    Wpath = PureWindowsPath(PATH)
    PATH = Path(Wpath)
    
    if os.path.exists(PATH / filename):
        base, ext = os.path.splitext(filename)
        counter = 0

        while os.path.exists(f"{base}_{counter}.csv"):
            counter += 1

        return f"{base}_{counter}.csv"
    
    else:
        return filename