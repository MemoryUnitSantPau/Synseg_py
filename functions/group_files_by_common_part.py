from functions.extract_common_part import extract_common_part 

def group_files_by_common_part(file_list, unique_texts, neuropil_channel):
    """
    Groups filenames by their common parts and sorts the groups based on neuropil_channel presence.

    Parameters:
    file_list (list of str): A list of filenames to be grouped.
    unique_texts (list of str): A list of unique texts found between underscores.
    neuropil_channel (str): The neuropil channel identifier.

    Returns:
    list of list of str: A list of grouped filenames, sorted based on neuropil_channel.
    """
    
    grouped_files = {}
    for filename in file_list:
        common_part = extract_common_part(filename)
        if common_part not in grouped_files:
            grouped_files[common_part] = [filename]
        else:
            grouped_files[common_part].append(filename)

    for common_part, filenames in grouped_files.items():
        filenames.sort(key=lambda x: (neuropil_channel in x, x))

    return list(grouped_files.values())