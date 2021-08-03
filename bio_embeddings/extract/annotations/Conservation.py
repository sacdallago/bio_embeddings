
from enum import Enum


class Conservation(Enum):
    cons_1 = "1"
    cons_2 = "2"
    cons_3 = "3"
    cons_4 = "4"
    cons_5 = "5"
    cons_6 = "6"
    cons_7 = "7"
    cons_8 = "8"
    cons_9 = "9"

    def __str__(self):
        return {
            # one-based to comply to ConSurfDB indexing
            self.cons_1: "Variable. Score=1 (1: variable, 9: conserved)",
            self.cons_2: "Variable. Score=2 (1: variable, 9: conserved)",
            self.cons_3: "Variable. Score=3 (1: variable, 9: conserved)",
            self.cons_4: "Mixed. Score=4 (1: variable, 9: conserved)",
            self.cons_5: "Mixed. Score=5 (1: variable, 9: conserved)",
            self.cons_6: "Mixed. Score=6 (1: variable, 9: conserved)",
            self.cons_7: "Conserved. Score=7 (1: variable, 9: conserved)",
            self.cons_8: "Conserved. Score=8 (1: variable, 9: conserved)",
            self.cons_9: "Conserved. Score=9 (1: variable, 9: conserved)",
        }.get(self)

    @staticmethod
    def isAAFeature():
        return True
