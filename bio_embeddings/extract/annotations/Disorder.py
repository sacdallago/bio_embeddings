from enum import Enum


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
    def isAAFeature():
        return True
