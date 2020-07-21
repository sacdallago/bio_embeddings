from enum import Enum


class Location(Enum):

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

    @staticmethod
    def isAAFeature():
        return False
