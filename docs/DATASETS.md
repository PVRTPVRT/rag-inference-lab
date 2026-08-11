# External dataset provenance

## BEIR / SciFact

The external retrieval evaluation downloads the BEIR-preprocessed SciFact archive
from the official BEIR host:

- URL: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`
- Published MD5: `5f7d1de60b170fc8027bb7898e2efca1`
- Evaluation split: BEIR `test`
- Expected scale: 5,183 documents and 300 test queries

The downloaded archive is stored under `.cache/beir/`, which is ignored by Git.
No SciFact corpus text, claims, or annotations are redistributed in this
repository. Only aggregate metrics and the top document IDs/scores needed to
audit a run are retained.

According to the original SciFact repository:

- claims and evidence annotations are CC BY 4.0;
- abstracts originate from S2ORC and are ODC-By 1.0;
- SciFact code is Apache 2.0.

The BEIR project separately notes that users remain responsible for respecting
each source dataset's license and attribution requirements.

Primary references:

- BEIR dataset list: <https://github.com/beir-cellar/beir/wiki/Datasets-available>
- SciFact repository: <https://github.com/allenai/scifact>
- SciFact license: <https://github.com/allenai/scifact/blob/master/LICENSE.md>
- BEIR paper: <https://openreview.net/forum?id=wCu6T5xFjeJ>
