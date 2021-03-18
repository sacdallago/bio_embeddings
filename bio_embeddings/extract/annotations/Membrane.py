from enum import Enum


class Membrane(Enum):

    MEMBRANE = 'Membrane bound'
    SOLUBLE = 'Soluble'
    UNKNOWN = "?"

    def __str__(self):
        return {
            self.MEMBRANE: "Membrane bound",
            self.SOLUBLE: "Soluble",
            self.UNKNOWN: "Unknown"
        }.get(self)

    @staticmethod
    def isAAFeature():
        return False
