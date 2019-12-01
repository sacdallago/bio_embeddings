from bio_embeddings.extract_features.features import FeatureInterface


class SecondaryStructure(FeatureInterface):

    def __init__(self):
        super().__init__()
        self._DSSP8 = None
        self._DSSP3 = None

    def isAAFeature(self):
        return True

    def set_DSSP8(self, DSSP8):
        """

        :param DSSP8: A string with equal length to the sequence containing the secondary structure in DSSP8 format (allowed chars: GHIBESTC)
        :return: void
        """

        #TODO: check that string only contains GHIBESTC

        self._DSSP8 = DSSP8

        pass

    def get_DSSP8(self):
        return self._DSSP8

    def set_DSSP3(self, DSSP3):
        """

        :param DSSP3: A string with equal length to the sequence containing the secondary structure in DSSP3 format (allowed chars: HEC)
        :return: void
        """

        #TODO: check that string only contains HEC

        self.DSSP3 = DSSP3

        pass

    def get_DSSP3(self):
        return self.DSSP3
