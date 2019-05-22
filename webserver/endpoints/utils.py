import re

# valid amino acid characters in input sequence
ACCEPTABLE_CHAR_TARGET_REGEX = "[^ACDEFGHIKLMNPQRSTVWY]"
# valid amino acid characters in any other sequence (for now, include B, X, ...)
ACCEPTABLE_CHARS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
    'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'
]


def check_valid_sequence(sequence):

    m = re.search(
        ACCEPTABLE_CHAR_TARGET_REGEX, sequence, re.IGNORECASE
    )
    if m:
        return False

    return True
