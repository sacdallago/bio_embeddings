
from enum import Enum


class Conservation(Enum):
    # zero-based indexing for raw predictions
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
            self.cons_1: "variable_1",
            self.cons_2: "variable_2",
            self.cons_3: "variable_3",
            self.cons_4: "average_4",
            self.cons_5: "average_5",
            self.cons_6: "average_6",
            self.cons_7: "conserved_7",
            self.cons_8: "conserved_8",
            self.cons_9: "conserved_9"
        }.get(self)

    @staticmethod
    def isAAFeature():
        return True
