from functions.extract_common_part2 import extract_common_part

def group_files_by_common_part(file_list:list, 
                               unique_texts:list)->list:
    """
    Group a list of filenames by their common parts based on unique texts.

    This function takes a list of filenames and a list of unique texts enclosed in parentheses. It groups the filenames
    by their common parts, which are extracted using the unique texts.

    Parameters:
        file_list (list of str): A list of filenames to be grouped.
        unique_texts (list of str): A list of unique texts enclosed in parentheses.

    Returns:
        list of list of str: A list of lists where each inner list contains filenames that share the same common part.

    Example:
        >>> file_list = ["Sample (text1) File1.tif", "Sample (text1) File2.tif", "Another (text2) File.tif"]
        >>> unique_texts = ["text1", "text2"]
        >>> group_files_by_common_part(file_list, unique_texts)
        [['Sample (text1) File1.tif', 'Sample (text1) File2.tif'], ['Another (text2) File.tif']]
    """
    grouped_files = {}
    for filename in file_list:
        common_part = extract_common_part(filename, unique_texts)
        if common_part not in grouped_files:
            grouped_files[common_part] = [filename]
        else:
            grouped_files[common_part].append(filename)
    return list(grouped_files.values())
