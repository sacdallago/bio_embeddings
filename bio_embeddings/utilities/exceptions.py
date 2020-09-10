class NoEmbeddingException(Exception):
    """
    Exception to handle the case no embedding has been computed, and one requests features.
    """


class MissingParameterError(Exception):
    """
    Exception for missing parameters
    """


class InvalidParameterError(Exception):
    """
    Exception for invalid parameter settings
    """


class SequenceTooLongException(Exception):
    """
    Exception when trying to embed sequences longer then the limit
    """


class MD5ClashException(Exception):
    """
    When remapping sequences from a fasta file, if there is an MD5 clash, this will stop the execution
    """


class TooFewComponentsException(InvalidParameterError):
    """
    Thrown when n_components is nonsensical (e.g. < 2)
    """


class ConversionUniqueMismatch(Exception):
    """
    Thrown when trying to remap using a mapping file which doesn't have as many uniuqye original_ids as md5 hashes
    """


class UnrecognizedEmbeddingError(Exception):
    """
    Thrown when trying to access embeddings for sequences which have no embedding.
    """


class InvalidAnnotationFileError(Exception):
    """
    Thrown when an annotation file contains invalid values
    """