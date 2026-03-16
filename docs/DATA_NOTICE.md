# Data Notice

This project expects GEO-derived files to be placed under a local `data/` directory.

Source website: NCBI Gene Expression Omnibus (GEO).

Reference dataset pages:

- `GSE33000`: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33000`
- `GSE122063`: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122063`
- `GSE109887`: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109887`

Expected layout:

```text
data/
|-- GSE33000mx/
|-- GSE122063yz1/
`-- GSE109887yz2/
```

Minimum training-cohort inputs:

- `data/GSE33000mx/geneMatrix.txt`
- `data/GSE33000mx/clinical.xlsx`

Minimum external-cohort inputs for each validation dataset:

- `geneMatrix.txt`
- `s1.txt`
- `s2.txt`

The repository uses historical local folder aliases (`GSE33000mx`, `GSE122063yz1`, `GSE109887yz2`). If downloaded GEO files use different names, reorganize them into the expected layout before running the pipeline.

The publish-ready repository does not depend on committed raw data files. Once the local data is available, the pipeline reads from `data/` and writes generated artifacts to `results/` automatically.
