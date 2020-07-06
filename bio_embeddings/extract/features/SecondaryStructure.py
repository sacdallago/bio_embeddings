from bio_embeddings.extract.features import FeatureInterface


class SecondaryStructure(FeatureInterface):
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

    def isAAFeature(self):
        return True
