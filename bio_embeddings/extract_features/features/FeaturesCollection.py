from bio_embeddings.extract_features.features import Disorder
from bio_embeddings.extract_features.features import Location
from bio_embeddings.extract_features.features import Membrane
from bio_embeddings.extract_features.features import SecondaryStructure


class FeaturesCollection:

    def __init__(self):
        self.secondaryStructure: SecondaryStructure = None
        self.disorder: Disorder = None
        self.location: Location = None
        self.membrane: Membrane = None

    def to_dict(self):
        result = {}

        if self.location is not None:
            result['predictedSubcellularLocalizations'] = self.location.get_location()
            pass
        if self.membrane is not None:
            result['predictedMembrane'] = self.membrane.to_stirng()
            pass
        if self.secondaryStructure is not None:
            result['predictedDSSP3'] = self.secondaryStructure.get_DSSP3()
            result['predictedDSSP8'] = self.secondaryStructure.get_DSSP8()
            pass
        if self.disorder is not None:
            result['predictedDisorder'] = self.disorder.get_disorder()
            pass

        return result
