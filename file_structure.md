CoRR stands for the Consortium for Reliability and Reproducibility, and its defining feature — the thing that makes it different from the other INDI datasets — is that it's built around test-retest. 
The same people were scanned on two or more separate occasions, on purpose.


In your connectome context, the mapping is almost certainly: nodes = brain regions (parcels from some atlas), 
and e_weight = functional connectivity between two regions — typically the Pearson correlation between
their BOLD time series. So n8–n87 = 0.133 means weak coupling; n10–n28 = 0.543 means stronger coupling. 
Whether it's undirected depends on the edgedefault in the <graph> tag, but functional connectivity is symmetric, 
so it'll almost certainly say undirected (each pair appears once).

But the edges aren't physical connections. 
A weight like 0.133 is a statistical relationship — how correlated two regions' activity time series are over the scan. 
Two regions can be strongly connected functionally without any direct physical wiring between them.

Age spans roughly 6 to 88 years. The youngest come from pediatric sites — IPCAS 7 covers 6–17 (mean 11.6) and NYU 2 spans 6.47–55.03 — and the oldest from the aging samples, with Munich's yearly aging sample (LMU 3) at 59–88 (mean 69.8) and Montreal (UM 1) at 55–84 (mean 65.4). Harvard University + 2
Also maybe race/ethnicity varition?

SiteNAge range (mean)% FemaleRetest designSWU 423517–27 (20)49between-session, ~302 dNYU 21876.5–55 (20.2)38hybrid, multi-retestUPSM 110010–19.7 (15.1)48developmentalUM 1 (Montreal)8055–84 (65.4)273 retestsNKI 12419–60 (34.4)7514 dUtah 1268–39 (20.2)0longitudinal, ~2.5 yr


That last point is the one most relevant to your world: this is genuinely a signal-reliability dataset, with the noise left in on purpose and quantified. If you wanted to demonstrate a denoising or weak-signal method's robustness under repeated measurement, the motion-corrupted scans plus the test-retest structure give you a built-in ground truth for "did the signal survive."

sub-0025001/
├── session_1/
│   ├── anat/  sub-0025001_ses-1_T1w.nii.gz       # structural
│   └── func/  sub-0025001_ses-1_task-rest_bold.nii.gz   # resting-state fMRI
├── session_2/
│   ├── anat/  sub-0025001_ses-2_T1w.nii.gz
│   └── func/  sub-0025001_ses-2_task-rest_bold.nii.gz