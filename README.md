# dispersion-ML
Code for Extracting Dispersion Curves From Ambient Noise Correlations Using Deep Learning

Paper at https://ieeexplore.ieee.org/document/9099269

Code is provided as-is **and not maintained**; filenames should roughly explain the purpose of the scripts. Data cannot be provided due to licensing agreements.

Recommended environment is Anaconda 3.5+ (Spyder IDE) and at least RTX 2060. Code was developed locally on RTX 2080ti so adjust for memory use accordingly.

Overall workflow (some scripts were needed for figure generation only): **makedata -> data_to_npy -> trainlb -> other evaluation scripts**
