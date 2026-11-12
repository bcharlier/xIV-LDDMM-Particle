xmodmap: Cross modality mapping
===============================

Geometric registration for Spatial Omics Data
---------------------------------------------

``xmodmap`` estimates geometric and functional mappings between spatial
datasets, including data acquired with different modalities and at different
scales. Each dataset is represented as a set of particles carrying spatial
positions and features, which makes the framework agnostic to the underlying
measurement. Supported data types include:

- spatial transcriptomics, from spot-based (e.g. Visium) to single-cell
  resolution (e.g. MERFISH, BARseq) measurements;
- stained tissue sections, such as histology images (e.g. H&E) and
  fluorescence microscopy;
- volumetric or planar imaging, such as MRI and reference atlases.

It supports rigid, affine and non-rigid registration. Applications include
mapping tissue-scale atlases to cellular and transcriptional data.

References
----------

Some of the methods used here are described in the following papers:

1. K. M. Stouffer, A. Trouvé, L. Younes, M. Kunst, L. Ng, H. Zeng, M. Anant, J. Fan, Y. Kim, X. Chen, M. Rue, and M. I. Miller. *Cross-modality mapping using image varifolds to align tissue-scale atlases to molecular-scale measures with application to 2d brain sections*. Nat Commun 15, 3530 (2024). https://doi.org/10.1038/s41467-024-47883-4
2. K. M. Stouffer, X. Chen, H. Zeng, B. Charlier, L. Younes, A. Trouvé, M. I. Miller. *xIV-LDDMM Toolkit: A Suite of Image-Varifold Based Technologies for Representing and Mapping 3D Imaging and Spatial-omics Data Simultaneously Across Scales*. Commun Biol 8, 1401 (2025). https://doi.org/10.1038/s42003-025-08800-7

Authors
-------

- Benjamin Charlier (benjamin.charlier@inrae.fr)
- Kaitlin Stouffer (kstouff4@jhmi.edu)
- Alain Trouvé (alain.trouve@ens-paris-saclay.fr)
- Dasun Premathilaka (hpremat1@jh.edu)
- Yi Xie (yxie91@jh.edu)

License
-------

`MIT License <https://www.mit.edu/~amini/LICENSE.md>`_


.. toctree::
   :maxdepth: 2
   :caption: xmodmap

   _auto_examples/index