import re

def extract_common_part(filename:str, 
                        unique_texts:list)->str:
    """
    Extract the common part of a filename based on unique texts.

    This function takes a filename and a list of unique texts and attempts to extract the common part of the
    filename that precedes any of the unique texts enclosed in parentheses.

    Parameters:
        filename (str): The filename to extract the common part from.
        unique_texts (list of str): A list of unique texts enclosed in parentheses.

    Returns:
        str or None: The common part of the filename if found, or None if no match is found.

    Example:
        >>> filename = "Sample (text1) Filename.tif"
        >>> unique_texts = ["text1", "text2"]
        >>> extract_common_part(filename, unique_texts)
        'Sample'
    """

    pattern = r'(.+) \((' + '|'.join(unique_texts) + r')\)\.tif'
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    return None
