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

## BEIR / ArguAna

The second external evaluation uses the official BEIR-preprocessed ArguAna
archive:

- URL: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip`
- Published MD5: `8ad3e3c2a5867cdced806d6503f29b99`
- Evaluation split: BEIR `test`
- Expected scale: about 8,670 documents and 1,406 test queries
- Task: retrieve the best counterargument to a query argument

The BEIR Hugging Face dataset card labels ArguAna as CC BY-SA 4.0. The original
artifact is attributed to Wachsmuth, Syed, and Stein. BEIR notes that its
preprocessing and software license do not replace source-dataset terms, so the
repository does not redistribute raw ArguAna text or labels.

ArguAna query arguments also occur in the corpus. The evaluation removes a
document whose ID equals the query ID before truncating to top 100; otherwise a
retriever can receive an invalid benefit from returning the query itself.

Dataset-selection reasoning and the protocol frozen before running results are
recorded in [`DATASET_SELECTION.md`](DATASET_SELECTION.md).

Primary references:

- BEIR dataset list: <https://github.com/beir-cellar/beir#available-datasets>
- BEIR ArguAna dataset card: <https://huggingface.co/datasets/BeIR/arguana>
- Original ArguAna artifact: <https://doi.org/10.5281/zenodo.3973258>
