# Data Notice

This project expects GEO-derived files to be placed under a local `data/` directory.

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

The publish-ready repository does not depend on committed raw data files. Once the local data is available, the pipeline reads from `data/` and writes generated artifacts to `results/` automatically.
