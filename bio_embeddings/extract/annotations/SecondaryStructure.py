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

    def __str__(self):
        return {
            self.ALPHA_HELIX: "α-helix",
            self.ISOLATED_BETA_BRIDGE: "Residue in isolated β-bridge",
            self.EXTENDED_STRAND: "Extended strand, participates in β ladder",
            self.THREE_HELIX: "3-helix",
            self.FIVE_HELIX: "5 helix",
            self.TURN: "Hydrogen bonded turn",
            self.BEND: "Bend",
            self.IRREGULAR: "Loop/Irregular",
            self.UNKNOWN: "Unknown"
        }.get(self)

    @staticmethod
    def isAAFeature():
        return True
