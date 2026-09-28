# Clinical _Escherichia coli_ strains with whole genome sequencing data and antimicrobial susceptibility meta-data for machine learning

## Author information

Alessandro Gerada

email: alessandro.gerada@liverpool.ac.uk

github: @agerada

## License

This work is licensed under CC BY 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/

# Introduction

This is a dataset of whole genome sequencing data (WGS) for 762 _Escherichia coli_ isolates with associate antimicrobial susceptibility testing (AST) meta-data. The dataset was generated from clinical strains retrieved from Liverpool Clinical Laboratories, UK (Liverpool University Hospitals NHS Trust). Isolates were originally isolated between 2017--2021. The dataset was produced to train machine learning models to predict antimicrobial minimum inhibitory concentration from WGS data. The isolates were chosen based on their AST phenotype, to generate a representative sample of important resistance mechanisms. In other words, this is not a representative random sample of _E. coli_ AST phenotypes --- rare phenotypes were over-sampled; common phenotypes were under-sampled.

# Contents

1. Whole genome sequence data - `genomes.zip`. Each file is in `.fna` format, with the filename corresponding to the strain number.
2. Annotated antimicrobial resistance (AMR) determinants - `annots.zip`. Each file is a `.tsv` of resistance determinants.
3. Antimicrobial susceptibility meta-data (disk diffusion and minimum inhibitory concentration) for strains in two formats:
  - `meta_data_bv_brc_format.txt` - long data format, compatible with the format used by PATRIC database on BV-BRC, see https://www.bv-brc.org/docs/quick_references/ftp.html for more,
  - `meta_data_tidy.txt` - wide data format, compatible with the `R` [`AMR`](https://msberends.github.io/AMR/).

# Data generation

- The `.fna` files were generated from Illumina short reads, assembled using `SPAdes` assembler (v3.15.4), using default parameters plus `--careful` and `--only-assembler` options.
- AMR resistance determinants were annotated using `AMRFinderPlus` version 3.11.14, using the default settings and database version `2023-07-13.2`.
- Disk diffusion testing was measured in the clinical laboratory at Liveprool Clinical Laboratories using EUCAST zones.
- The `genome_name` includes information around additional phenotypic testing performed in the clinical laboratory, specifically - `(ESBL+)` = extended-spectrum beta-lactamase present (usually on disk diffusion synergy); `(CARB+)` = carbapenemase present (usually on molecular testing).
- Minimum inhibitory concentrations were measured in the [APT](https://www.liverpool.ac.uk/apt) research laboratory at the University of Liverpool using agar dilution.

# Data dictionary

## `meta_data_bv_brc_format.txt`

Field                      | Data type | Description
---------------------------|-----------|--------------------------------------
`genome_id`                | string    | Unique identifier for strains
`genome_name`              | string    | Taxonomic organism name
`antibiotic`               | string    | Antimicrobial name
`resistant_phenotype`      | string    | S/I/R (or NA for MIC measurements)
`laboratory_typing_method` | string    | Disk diffusion or Agar dilution
`measurement`              | string    | MIC values, including ">" and "≤" (or NA for disk diffusion measurements)
`measurement_unit`         | string    | "mg/L" (or NA for disk diffusion measurements)

Note: Disk diffusion results are only provided as S/I/R values as zone sizes were not available. On the other hand, resistant phenotypes are not provided for MIC results -- please use the `MIC` package (or alternative) to generate S/I/R if required.

## `meta_data_tidy.txt`

Field                      | Data type | Description
---------------------------|-----------|-------------------------------------
`genome_id`                | string    | Unique identifier for strains
`genome_name`              | string    | Taxonomic organism name
`disk_*`                   | string    | S/I/R (or NA)
`mic_*`                    | string    |  MIC values, including ">" and "≤" (or NA for disk diffusion measurements)

Note: Data is provided in a wide format, `disk_*` columns have disk susceptibility phenotype, with the second part representing the WHO code for each antimicrobial, e.g., `disk_AMC` = disk susceptibility for amoxicillin/clavulanic acid; `disk_TZP` = disk susceptibility for piperacillin/tazobactam, etc. The same pattern is observed for `mic_*` values.

# Acknowledgements

This work was funded, in part, by UKRI Doctoral Training Program [grant ref: 2599501] and the Wellcome Trust [grant ref: 226691/Z/22/Z].

We would like to thank: 

- Centre for Genomics Research, University of Liverpool, for sequencing support,
- Liverpool Clinical Laboratories for availability of strains,
- Ms Valerie Price for support in retrieving strains from storage.
