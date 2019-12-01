class NoEmbeddingException(Exception):
    """
    Exception to handle the case no embedding has been computed, and one requests features.
    """


class CannotInferModelVersionException(Exception):
    """
    Exception gets risen when a version is supplied but also files that are not relevant to that version
    (e.g. version = 1 + vocabulary file).
    """