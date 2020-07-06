from bio_embeddings.extract.features import FeatureInterface, InvalidFeatureException


class Membrane(FeatureInterface):

    MEMBRANE = 'Membrane bound'
    SOLUBLE = 'Soluble'
    UNKONWN = "?"

    def isAAFeature(self):
        return False
