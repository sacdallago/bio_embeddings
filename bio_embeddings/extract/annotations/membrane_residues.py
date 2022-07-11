from enum import Enum


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

