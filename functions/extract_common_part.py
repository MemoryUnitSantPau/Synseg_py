def extract_common_part(filename):
    """
    Extracts the common part of a filename up to the first underscore.

    Parameters:
    filename (str): The filename from which to extract the common part.

    Returns:
    str: The extracted common part of the filename.
    """

    return filename.split("_")[0]
