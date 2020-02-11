from bio_embeddings.extract_features.features import FeatureInterface, InvalidFeatureException


class Membrane(FeatureInterface):

    def __init__(self):
        super().__init__()
        self._membrane = None

    def isAAFeature(self):
        return False

    def set_membrane(self, is_membrane):
        """
        :param is_membrane: A Bool setting if membrane or not.
        :return: void
        """

        # TODO: check that string only equals accepted locations

        if not isinstance(is_membrane, bool):
            raise InvalidFeatureException

        self._membrane = is_membrane

        pass

    def is_membrane(self):
        return self._membrane

    def to_stirng(self):
        if self._membrane:
            return "Membrane bound"
        else:
            return "Soluble"

