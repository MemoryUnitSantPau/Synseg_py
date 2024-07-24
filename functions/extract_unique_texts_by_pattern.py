import re

def extract_unique_texts_by_pattern(strings_list):
        """
        Extract unique text segments enclosed within parentheses from a list of strings.

        This function takes a list of strings and searches for text segments that are enclosed
        within parentheses. It extracts these segments and returns a list of unique text segments.

        Args:
            strings_list (list): A list of strings containing text segments enclosed within parentheses.

        Returns:
            list: A list of unique text segments extracted from the input strings.

        Example:
            strings = ["(apple) pie", "A (banana) split", "Cherry (pie)"]
            unique_texts = extract_unique_texts_between_parentheses(strings)
            print(unique_texts)  # Output: ['apple', 'banana', 'pie']

        Note:
            - The function uses regular expressions to extract text between parentheses.
            - Text segments within the same string are treated as separate matches.
            - The extracted text segments are unique, i.e., duplicates are removed.

        Reference:
            Regular expression pattern for extracting text between parentheses: r'\((.*?)\)'

        Warning:
            The function might not handle nested parentheses or complex patterns perfectly.
            It is designed for simple cases where text segments are directly enclosed in parentheses.
        """

        pattern = r'\((.*?)\)'
        matches = [re.findall(pattern, string) for string in strings_list]
        unique_texts = set([match for matches_list in matches for match in matches_list])
        return list(unique_texts)
