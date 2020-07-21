from enum import Enum


class Membrane(Enum):

    MEMBRANE = 'Membrane bound'
    SOLUBLE = 'Soluble'
    UNKONWN = "?"

    @staticmethod
    def isAAFeature():
        return False
