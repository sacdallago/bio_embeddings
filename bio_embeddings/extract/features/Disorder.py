from bio_embeddings.extract.features import FeatureInterface


class Disorder(FeatureInterface):
    DISORDER = 'X'
    ORDER = '-'
    UNKNOWN = '?'

    def isAAFeature(self):
        return True
