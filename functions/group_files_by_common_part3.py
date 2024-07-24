from functions.extract_common_part2 import extract_common_part

def group_files_by_common_part(file_list, unique_texts):
    """
    Group a list of filenames by their common parts without unique text.

    This function takes a list of filenames and a list of unique text segments that are enclosed
    within parentheses. It uses the `extract_common_part` function to remove the unique text
    from each filename and extract the common part. Then, it groups the filenames based on their
    common parts, forming a dictionary where keys are common parts and values are lists of filenames.

    Args:
        file_list (list): A list of filenames to group.
        unique_texts (list): A list of unique text segments that were enclosed in parentheses.

    Returns:
        list: A list of lists, where each inner list contains filenames with the same common part.

    Example:
        file_list = ["Document (apple).tif", "Manual (apple).tif", "Report (banana).tif"]
        unique_texts = ['apple', 'banana']
        grouped_files = group_files_by_common_part(file_list, unique_texts)
        print(grouped_files)
        # Output: [['Document (apple).tif', 'Manual (apple).tif'], ['Report (banana).tif']]

    Note:
        - The function depends on the `extract_common_part` function for extracting common parts.
        - Filenames with the same common part are grouped together.
        - Filenames without a common part (no match) are treated as separate groups.

    Warning:
        The function's success relies on accurate unique texts and the filename pattern.
        It might not handle complex filename variations or patterns perfectly.

    See Also:
        - extract_common_part: A function used to extract common parts from filenames.

    Reference:
        extract_common_part function for extracting common parts of filenames.
    """
    grouped_files = {}
    for filename in file_list:
        common_part = extract_common_part(filename, unique_texts)
        if common_part not in grouped_files:
            grouped_files[common_part] = [filename]
        else:
            grouped_files[common_part].append(filename)
    return list(grouped_files.values())

