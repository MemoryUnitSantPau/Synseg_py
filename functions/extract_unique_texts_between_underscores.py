import re


def extract_unique_texts_between_underscores(strings_list):
    """
    Extracts unique texts found between underscores in a list of strings.

    Parameters:
    strings_list (list of str): A list of strings containing underscores.

    Returns:
    list of str: A list of unique texts found between underscores.
    """
    
    pattern = r'_(.*?)(?:_|\.tif)'
    matches = [re.findall(pattern, string) for string in strings_list]
    unique_texts = set([match for matches_list in matches for match in matches_list])
    return list(unique_texts)