# Reading AMRFinderPlus output

AMRFinderPlus (NCBI; Feldgarden *et al.* 2021 *Sci Rep* 11:12728) searches assemblies or proteins against the
Reference Gene Catalog and reports acquired resistance genes and, when run with `--organism`, known point
mutations. It plays the same role as ResFinder (acquired genes) plus PointFinder (point mutations) from DTU/CGE
(Bortolaia *et al.* 2020 *J Antimicrob Chemother* 75:3491; Zankari *et al.* 2017 *J Antimicrob Chemother* 72:2764).

## Key columns (one row per hit)

| Column | Meaning |
|---|---|
| `Gene symbol` | the determinant; point mutations are `gene_REFposALT`, e.g. `gyrA_S83L`, `parC_S80I` |
| `Element type` | `AMR`, `STRESS` (biocides, metals, heat) or `VIRULENCE` |
| `Element subtype` | `AMR` (acquired gene), `POINT` (point mutation), `BIOCIDE`, ... |
| `Class` / `Subclass` | drug class affected, e.g. `QUINOLONE`; multi-drug hits list several |
| `Method` | how it was found: `ALLELEX` / `EXACTX` (exact allele), `BLASTX` (close match), `PARTIALX` / `PARTIAL_CONTIG_ENDX` (partial, often at a contig edge), `POINTX` (point mutation), `HMM` (distant family match) |
| `% Coverage` / `% Identity` | alignment quality against the reference sequence |

## Pitfalls when turning calls into features

- **Point mutations need `--organism`**: without it (for *E. coli*, `--organism Escherichia`) no `gyrA` or
  `parC` mutations are reported at all.
- **Naming granularity**: an exact allele gives a specific symbol (`blaCTX-M-15`), a partial or distant hit a
  family symbol (`blaCTX-M`, `blaTEM`). The same gene can therefore appear under two names in different
  isolates; grouping allele families is a reasonable feature-engineering step.
- **Partial hits** at contig ends come from fragmented assemblies; they are usually real genes, but coverage is incomplete.
- **Absence is not certainty**: a missing call can be a real absence, a novel allele, or a gap in the assembly.
- **Multi-class hits** (`marR`, `acrR`, `soxR` variants) affect several drugs through efflux; their
  fluoroquinolone effect is usually modest.
- **Rare features**: determinants carried by very few isolates cannot be estimated well; pool them (by family
  or mechanism) or let a sparse prior shrink them.
