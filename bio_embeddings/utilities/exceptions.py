class NoEmbeddingException(Exception):
    """
    Exception to handle the case no embedding has been computed, and one requests features.
    """


class CannotInferModelVersionException(Exception):
    """
    Exception gets risen when a version is supplied but also files that are not relevant to that version
    (e.g. version = 1 + vocabulary file).
    """


class MissingParameterError(Exception):
    """
    Exception for missing parameters
    """


class InvalidParameterError(Exception):
    """
    Exception for invalid parameter settings
    """

class CannotFindDefaultFile(Exception):
    """
    Exception for invalid file download request
    """
