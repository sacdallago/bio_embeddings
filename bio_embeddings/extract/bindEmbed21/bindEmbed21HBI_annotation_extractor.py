import logging
import numpy

from Bio import SeqIO
from pathlib import Path
import collections

from typing import List
from enum import Enum

from bio_embeddings.extract.annotations import BindingResidues
from bio_embeddings.utilities import get_model_directories_from_zip

logger = logging.getLogger(__name__)

# Label mappings
_metal_labels = {
    -1: BindingResidues.not_inferred,
    0: BindingResidues.non_binding,
    1: BindingResidues.metal
}

_nucleic_labels = {
    -1: BindingResidues.not_inferred,
    0: BindingResidues.non_binding,
    1: BindingResidues.nucleic_acid
}

_small_molecules_labels = {
    -1: BindingResidues.not_inferred,
    0: BindingResidues.non_binding,
    1: BindingResidues.small_molecule
}


BasicBindingResidueResult = collections.namedtuple('BasicBindingResidueResult',
                                                   'metal_ion nucleic_acids small_molecules')


class BindEmbed21HBIAnnotationExtractor:
    """
    Extract binding predictions for 3 different ligand classes (metal ions, nucleic acids, small molecules)
    using homology-based inference.
    """
    necessary_directories = ["annotations_directory"]

    def __init__(self, **kwargs):
        """
        Initialize annotation extractor. Must define non-positional arguments for paths of files.
        """

        self._options = kwargs

        # Download the annotation files if needed
        for directory in self.necessary_directories:
            if not self._options.get(directory):
                self._options[directory] = get_model_directories_from_zip(model=f"bindembed21hbi", directory=directory)

        metal_annotations_file_path = Path(self._options['annotations_directory']) / 'annotations_metal.fasta'
        nuc_annotations_file_path = Path(self._options['annotations_directory']) / 'annotations_nuc.fasta'
        small_annotations_file_path = Path(self._options['annotations_directory']) / 'annotations_small.fasta'

        metal_annotations_fasta = SeqIO.to_dict(SeqIO.parse(str(metal_annotations_file_path), 'fasta'))
        nuc_annotations_fasta = SeqIO.to_dict(SeqIO.parse(str(nuc_annotations_file_path), 'fasta'))
        small_annotations_fasta = SeqIO.to_dict(SeqIO.parse(str(small_annotations_file_path), 'fasta'))

        self.metal_annotations = self.convert_annotations_to_list(metal_annotations_fasta)
        self.nuc_annotations = self.convert_annotations_to_list(nuc_annotations_fasta)
        self.small_annotations = self.convert_annotations_to_list(small_annotations_fasta)

    @staticmethod
    def convert_annotations_to_list(fasta_annotations):
        annotations = collections.defaultdict(list)

        for k in fasta_annotations.keys():
            uni_id = fasta_annotations[k].id
            anno_seq = fasta_annotations[k].seq
            anno_list = []

            for a in anno_seq:
                if a == '-':  # not annotated as binding
                    anno_list.append(0)
                else:
                    anno_list.append(1)

            annotations[uni_id] = anno_list

        return annotations

    def get_binding_residues(self, hit: dict) -> BasicBindingResidueResult:
        indices_query = self._get_indices_seq(hit['qstart'], hit['qaln'])
        indices_target = self._get_indices_seq(hit['tstart'], hit['taln'])

        target_id = hit['target']
        inferred_binding = numpy.zeros([hit['qlen'], 3], dtype=numpy.float32) - 1

        for idx, pos2 in enumerate(indices_target):
            pos1 = indices_query[idx]
            if pos1 >= 1 and pos2 >= 1:  # both positions are aligned

                metal_anno = self.metal_annotations[target_id][pos2-1]
                nuc_anno = self.nuc_annotations[target_id][pos2-1]
                small_anno = self.small_annotations[target_id][pos2-1]

                inferred_binding[pos1] = [metal_anno, nuc_anno, small_anno]

        # Map raw class predictions (integers) to class labels (strings)
        pred_metal = self._class2label(_metal_labels, inferred_binding[0, ])
        pred_nuc = self._class2label(_nucleic_labels, inferred_binding[1, ])
        pred_small = self._class2label(_small_molecules_labels, inferred_binding[2, ])

        return BasicBindingResidueResult(metal_ion=pred_metal, nucleic_acids=pred_nuc, small_molecules=pred_small)

    @staticmethod
    def _get_indices_seq(start, seq):

        indices = []

        for i in range(0, len(seq)):
            if seq[i] == '-':  # position is gap
                indices.append(0)
            else:
                indices.append(start)
                start += 1

        return indices

    @staticmethod
    def _class2label(label_dict, yhat) -> List[Enum]:
        return [label_dict[el] for el in yhat]
