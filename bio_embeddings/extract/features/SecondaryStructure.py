from enum import Enum


class SecondaryStructure(Enum):
    # From https://swift.cmbi.umcn.nl/gv/dssp/

    ALPHA_HELIX = "H"
    ISOLATED_BETA_BRIDGE = "B"
    EXTENDED_STRAND = "E"
    THREE_HELIX = "G"
    FIVE_HELIX = "I"
    TURN = "T"
    BEND = "S"
    IRREGULAR = "C"
    UNKNOWN = "?"

    @staticmethod
    def isAAFeature():
        return True
