<!-- README.md -->
<!-- PROJECT DESCRIPTION -->
<!-- Mention the name "CleanMachine" -->
<!-- The app allows you to pass it a wave file containing noisy speech and denoise it using one of 3 algorithms. Spectral Subtraction, Wiener Filter or Machine Learning  -->
<!-- Mention the machine learning is implemented based on the paper  "Germain, Francois G., et al. Speech Denoising with Deep Feature Losses. arXiv:1806.10522, arXiv, 14 Sept. 2018. arXiv.org, https://doi.org/10.48550/arXiv.1806.10522." -->
<!-- And the other two algorithms come from techniques curated in "Loizou, Philipos C. Speech Enhancement: Theory and Practice. 2nd ed., CRC Press, 2013. DOI.org (Crossref), https://doi.org/10.1201/b14529." -->
<!-- RUN INSTRUCTIONS -->
<!-- mention steps on how to run the denoise -->
<!-- the  CLI entry point is src/cm.py  -->
<!-- Takes 2 arguments at the moment  -->
<!-- 1. '-f' a relative or absolute pathname to the input wave file -->
<!-- Mention that the input is a wave file -->
<!-- 2. '-a' algorithm type (SS, WF, or ML) used for the de-noising task. Spectral Subtraction, Wiener Filter, or Machine Learning respectively' -->
<!-- the next one is still on the todo -->
<!-- 3. '-o' output file name (optional) -->
<!-- Mention that ouput is currently saved with the same name as the input file with `XX_denoised` appended to it where XX refers the lower case version of the -a arg -->
<!-- Mention that the output is saved in the same directory as the input file -->
<!-- Mention that the output is a wave file -->
<!-- Mention that the individual modules can also be imported into other python project using `from traditional import WienerFilter, SpectralSubtraction` or `from infer import DeepDenoise` -->

<!-- FILE CONTENTS -->
<!-- "./src/cm.py" as mentioned this is the just entry point for the CLI. -->
<!-- "./src/traditional.py" This file contains 2 functions that use tradiontal methods for speech denoisng. Namely Wiener Filter and Spectral Subtraction -->