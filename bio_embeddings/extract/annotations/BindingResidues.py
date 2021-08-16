from enum import Enum


class BindingResidues(Enum):
    metal = "M"
    nucleic_acid = "N"
    small_molecule = "S"
    protein = "P"
    non_binding = "-"

    def __str__(self):
        return {
            self.metal: "Binding to metal ion",
            self.nucleic_acid: "Binding to DNA or RNA",
            self.small_molecule: "Binding to small (regular) molecule",
            self.protein: "Binding to protein",
            self.non_binding: "Not binding",
        }.get(self)

    @staticmethod
    def isAAFeature():
        return True
