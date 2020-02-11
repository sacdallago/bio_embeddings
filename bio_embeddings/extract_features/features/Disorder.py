from bio_embeddings.extract_features.features import FeatureInterface


class Disorder(FeatureInterface):

    def __init__(self):
        super().__init__()
        self._disorder = None

    def isAAFeature(self):
        return True

    def set_disorder(self, disorder):
        """

        :param disorder: A string with equal length to the sequence containing the disordered regions (allowed chars: -X)
        :return: void
        """

        #TODO: check that string only contains -X

        self._disorder = disorder

        pass

    def get_disorder(self):
        return self._disorder
