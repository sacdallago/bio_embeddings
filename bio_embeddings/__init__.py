"""
The functionality of bio_embeddings is split into 5 different modules

.. autosummary::
   bio_embeddings.embed
   bio_embeddings.extract
   bio_embeddings.project
   bio_embeddings.utilities
   bio_embeddings.visualize
"""

import bio_embeddings.embed
import bio_embeddings.extract
import bio_embeddings.project
import bio_embeddings.utilities
import bio_embeddings.visualize

__all__ = ["embed", "extract", "project", "utilities", "visualize"]
