from bio_embeddings.extract.features import FeatureInterface, InvalidFeatureException


class Location(FeatureInterface):

    CELL_MEMBRANE = 'Cell-Membrane'
    CYTOPLASM = 'Cytoplasm'
    ENDOPLASMATIC_RETICULUM = 'Endoplasmic reticulum'
    GOLGI_APPARATUS = 'Golgi - Apparatus'
    LYSOSOME_OR_VACUOLE = 'Lysosome / Vacuole'
    MITOCHONDRION = 'Mitochondrion'
    NUCLEUS = 'Nucleus'
    PEROXISOME = 'Peroxisome'
    PLASTID = 'Plastid'
    EXTRACELLULAR = 'Extra - cellular'
    UNKNOWN = '?'

    def isAAFeature(self):
        return False
