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

    def __str__(self):
        return {
            self.CELL_MEMBRANE: "Cell-Membrane",
            self.CYTOPLASM: "Cytoplasm",
            self.ENDOPLASMATIC_RETICULUM: "Endoplasmic reticulum",
            self.GOLGI_APPARATUS: "Golgi - Apparatus",
            self.LYSOSOME_OR_VACUOLE: "Lysosome / Vacuole",
            self.MITOCHONDRION: 'Mitochondrion',
            self.NUCLEUS: 'Nucleus',
            self.PEROXISOME: 'Peroxisome',
            self.PLASTID: 'Plastid',
            self.EXTRACELLULAR: 'Extra - cellular',
            self.UNKNOWN: '?'
        }.get(self)

    @staticmethod
    def isAAFeature():
        return False
