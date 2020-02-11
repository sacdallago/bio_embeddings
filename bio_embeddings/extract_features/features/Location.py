from bio_embeddings.extract_features.features import FeatureInterface, InvalidFeatureException


class Location(FeatureInterface):

    def __init__(self):
        super().__init__()
        self._location = None

    def isAAFeature(self):
        return False

    def set_location(self, location):
        """
        :param location: A string representing the protein's sub-cellular location. Accepted locations are:
                Cell-Membrane
                Cytoplasm
                Endoplasmic reticulum
                Golgi-Apparatus
                Lysosome/Vacuole
                Mitochondrion
                Nucleus
                Peroxisome
                Plastid
                Extra-cellular
        :return: void
        """

        # TODO: check that string only equals accepted locations

        if location not in ['Cell-Membrane', 'Cytoplasm', 'Endoplasmic reticulum', 'Golgi-Apparatus',
                     'Lysosome/Vacuole', 'Mitochondrion', 'Nucleus', 'Peroxisome', 'Plastid', 'Extra-cellular']:
            raise InvalidFeatureException

        self._location = location

        pass

    def get_location(self):
        return self._location

