# cosmos

## Paper pipeline (one command)

a) Setup:
   cd /Users/kurtbannister/code/cosmos
   source venv/bin/activate
   pip install -r requirements.txt

b) Dry-run:
   python3 paper_pipeline.py --dry-run
   (prints job matrix, number of seeds, total runs, output folder path)

c) Run:
   python3 paper_pipeline.py
   (default workers=4 on Mac)
   Or override workers:
   python3 paper_pipeline.py --workers 6

d) Resume after interruption:
   python3 paper_pipeline.py --resume --outdir <same_outdir>
   Explain: default creates new timestamp folder; to resume you must point to the same outdir.

e) Where outputs live:
   paper_outputs/<timestamp>/
     raw/
     tables/
     figures/
     REPORT.md
     pipeline.log

f) What to look at first:
   tables/job_summary.csv
   figures/Fig2_*.png
   REPORT.md

Makefile shortcut:

```bash
make paper
```