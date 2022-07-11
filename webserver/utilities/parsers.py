import json
from enum import Enum
from typing import NamedTuple, List, Union


class MembraneResidues(Enum):
    TMB_IN_OUT = 'B'
    TMB_OUT_IN = 'b'
    TMH_IN_OUT = 'H'
    TMH_OUT_IN = 'h'
    SIGNAL_PEPTIDE = 'S'
    NON_TRANSMEMBRANE = '.'

    def __str__(self):
        return {
            self.TMB_IN_OUT: 'Transmembrane beta strand (IN --> OUT orientation)',
            self.TMB_OUT_IN: 'Transmembrane beta strand (OUT --> IN orientation)',
            self.TMH_IN_OUT: 'Transmembrane alpha helix (IN --> OUT orientation)',
            self.TMH_OUT_IN: 'Transmembrane alpha helix (OUT --> IN orientation)',
            self.SIGNAL_PEPTIDE: 'Signal peptide',
            self.NON_TRANSMEMBRANE: 'Non-Transmembrane'
        }.get(self)

    @staticmethod
    def is_aa_feature():
        return True


class SecondaryStructure(Enum):
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

    def __str__(self):
        return {
            self.ALPHA_HELIX: "α-helix",
            self.ISOLATED_BETA_BRIDGE: "Residue in isolated β-bridge",
            self.EXTENDED_STRAND: "Extended strand, participates in β ladder",
            self.THREE_HELIX: "3-helix",
            self.FIVE_HELIX: "5 helix",
            self.TURN: "Hydrogen bonded turn",
            self.BEND: "Bend",
            self.IRREGULAR: "Loop/Irregular",
            self.UNKNOWN: "Unknown"
        }.get(self)

    @staticmethod
    def is_aa_feature():
        return True


class Disorder(Enum):
    DISORDER = 'X'
    ORDER = '-'
    UNKNOWN = '?'

    def __str__(self):
        return {
            self.DISORDER: "Disorder",
            self.ORDER: "Order",
            self.UNKNOWN: "Unknown"
        }.get(self)

    @staticmethod
    def is_aa_feature():
        return True


class BindingResidues(Enum):
    metal = "M"
    nucleic_acid = "N"
    small_molecule = "S"
    protein = "P"
    non_binding = "-"
    not_inferred = "0"

    def __str__(self):
        return {
            self.metal: "Binding to metal ion",
            self.nucleic_acid: "Binding to DNA or RNA",
            self.small_molecule: "Binding to small (regular) molecule",
            self.protein: "Binding to protein",
            self.non_binding: "Not binding",
            self.not_inferred: "Not inferred"
        }.get(self)

    @staticmethod
    def is_aa_feature():
        return True


class Source(NamedTuple):
    url: str
    id: str
    name: str = "BioEmbeddings"

    def toDict(self):
        """
        url: the URL of where the prediction can be retrieved
        id: the ID of the job
        name: name of the source
        """
        return {
            "url": self.url,
            "id": self.id,
            "name": self.name
        }

    def toJSON(self):
        return json.dumps(self.toDict(), sort_keys=True, indent=4)


class Evidence(NamedTuple):
    source: Source
    code: str = "ECO:0007669"

    def toDict(self):
        """
        source: the source from where the annotation was taken
        code: the evidence code, usually ECO:XXXXXX, from the evidence code ontology: evidenceontology.org
        """
        return {
            "source": self.source.toDict(),
            "code": self.code
        }

    def toJSON(self):
        return json.dumps(self.toDict(), sort_keys=True, indent=4)


_HEX_COLORS = {
    "HEX_COL1": '#648FFF',
    "HEX_COL2": '#785EF0',
    "HEX_COL3": '#DC267F',
    "HEX_COL4": '#FE6100',
    "HEX_COL5": '#FFB000',

    # These colors shouldn't be used unless absolutely necessary!
    "HEX_COL6": '#FFF000',
    "HEX_COL7": '#FF3000',
    "HEX_COL8": '#FEA100',
}

_FEATURE_COLORS = {
    Disorder.DISORDER: _HEX_COLORS['HEX_COL3'],
    Disorder.ORDER: _HEX_COLORS['HEX_COL1'],

    SecondaryStructure.ALPHA_HELIX: _HEX_COLORS['HEX_COL1'],
    SecondaryStructure.EXTENDED_STRAND: _HEX_COLORS['HEX_COL3'],
    SecondaryStructure.IRREGULAR: _HEX_COLORS['HEX_COL5'],

    SecondaryStructure.ISOLATED_BETA_BRIDGE: _HEX_COLORS['HEX_COL2'],
    SecondaryStructure.TURN: _HEX_COLORS['HEX_COL4'],
    SecondaryStructure.BEND: _HEX_COLORS['HEX_COL8'],

    SecondaryStructure.THREE_HELIX: _HEX_COLORS['HEX_COL6'],
    SecondaryStructure.FIVE_HELIX: _HEX_COLORS['HEX_COL7'],
}


class ProtVistaFeature(NamedTuple):
    begin: int
    end: int
    evidences: List[Evidence]
    description: Union[SecondaryStructure, Disorder]
    type: str
    category: str = "BIO_EMBEDDINGS"

    def toDict(self):
        """
        description: the description of the feature at that position, e.g. "Turn" for secondary structure
        category: general category of the feature
        evidences: list of evidences for the feature
        color: the HEX color to use in the feature viewer
        type: the type of feature, including prediction method, e.g. "SECONDARY_STRUCTURE_(SEQVEC)"
        beginning: the start of the feature in UniProt numbering (sequence starts at 1)
        end: the end of the feature in UniProt numbering (sequence starts at 1)
        """
        return {
            "description": str(self.description),
            "begin": self.begin,
            "end": self.end,
            "evidences": [e.toDict() for e in self.evidences],
            "type": self.type,
            "color": _FEATURE_COLORS.get(self.description),
            "category": self.category
        }

    def toJSON(self):
        return json.dumps(self.toDict(), sort_keys=True, indent=4)


# Note: this function is O(n), with n = length of the SeqRecord
def annotations_to_protvista_converter(features_string: str, evidences: List[Evidence], type: str, feature_enum) -> \
List[ProtVistaFeature]:
    features = list()

    current = None

    for i, AA_feature in enumerate(list(features_string)):
        AA_annotation = feature_enum(AA_feature)

        if not current:
            current = ProtVistaFeature(
                description=AA_annotation,
                begin=i + 1,
                end=i + 1,
                evidences=evidences,
                type=type
            )
        elif current.description != AA_annotation:
            features.append(current.toDict())
            current = ProtVistaFeature(
                description=AA_annotation,
                begin=i + 1,
                end=i + 1,
                evidences=evidences,
                type=type
            )
        else:
            current = current._replace(end=i + 1)

    features.append(current.toDict())

    return features
