def check_continuity(column:list) -> int:
    """
    Count the consecutive elements of a list containing 1 or true.
    """
    consecutives = 0
    for element in column:
        if element == 1 or element is True:
            consecutives += 1
            if consecutives >=2:
                return 1
        elif element == 0:
            consecutives = 0
    return consecutives