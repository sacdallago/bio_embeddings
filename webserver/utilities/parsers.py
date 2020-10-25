import json

from Bio import SeqRecord
from typing import NamedTuple, List, Union

from bio_embeddings.extract.annotations import SecondaryStructure, Disorder


class Source(NamedTuple):
    url: str
    id: str
    name: str = "BioEmbeddings"

    def toDict(self):
        """
        url: the URL of where the prediction can be retreived
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
}


_FEATURE_COLORS = {
    Disorder.DISORDER: _HEX_COLORS['HEX_COL3'],
    Disorder.ORDER: _HEX_COLORS['HEX_COL1'],
    SecondaryStructure.ALPHA_HELIX: _HEX_COLORS['HEX_COL1'],
    SecondaryStructure.EXTENDED_STRAND: _HEX_COLORS['HEX_COL3'],
    SecondaryStructure.IRREGULAR: _HEX_COLORS['HEX_COL5'],
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
def seqrecord_to_features(FASTA: SeqRecord, evidences: List[Evidence], type: str, feature_enum) -> List[ProtVistaFeature]:
    features = list()

    current = None

    for i, AA_feature in enumerate(list(FASTA.seq)):
        AA_annotation = feature_enum(AA_feature)

        if not current:
            current = ProtVistaFeature(
                description=AA_annotation,
                begin=i+1,
                end=i+1,
                evidences=evidences,
                type=type
            )
        elif current.description != AA_annotation:
            features.append(current)
            current = ProtVistaFeature(
                description=AA_annotation,
                begin=i+1,
                end=i+1,
                evidences=evidences,
                type=type
            )
        else:
            current.end = i+1

    features.append(current)

    return features
