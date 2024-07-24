import re

def extract_unique_texts_between_parentheses(strings_list:list)->list:
    """
    Extract unique texts found between parentheses in a list of strings.

    This function searches for text enclosed within parentheses in a list of strings and returns a list of unique
    texts found.

    Parameters:
        strings_list (list of str): A list of strings to search for text between parentheses.

    Returns:
        list of str: A list of unique texts found between parentheses in the input strings.

    Example:
        >>> strings_list = ["(text1)", "Sample (text2) string (text1) with multiple (text2) occurrences"]
        >>> extract_unique_texts_between_parentheses(strings_list)
        ['text2', 'text1']
    """

    pattern = r'\((.*?)\)'
    matches = [re.findall(pattern, string) for string in strings_list]
    unique_texts = set([match for matches_list in matches for match in matches_list])
    return list(unique_texts)