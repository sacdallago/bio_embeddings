from enum import Enum


class Disorder(Enum):
    DISORDER = 'X'
    ORDER = '-'
    UNKNOWN = '?'

    @staticmethod
    def isAAFeature():
        return True
