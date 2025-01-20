# oessa

Offline EEG state space analysis (non-GUI version)
<br/>(For GUI version that supports online and offline analysis, see this [repository](https://github.com/VBits/EEGspace))

Input files: 
<br/>1) .smrx files (Spike2 file format) 
<br/>2) or alternatively binary files (eg. mat files)
<br/>
<br/>
* A spectrum from the raw EEG signal is calculated using the Multitaper method. The resolution can be as low as 2 seconds per epoch (default).<br/>
  * 3 dataframes will be saved. The multitaper spectrum, a smoothed version of the spectrum (nonlinear smoothing using a median filter to better preserve transitions), and a normalized spectrum (lowest quantile power in each bin is subtracted)<br/>
* The data is then transformed into a low dimensional space using an LDA previously trained on data from multiple B6Jv animals<br/>
  * Alternatively temporary state labels from a neural network can be generated and used to train a new LDA<br/>
* A density based method of clustering is applied in low dimensional space to a subset of data. 4 states can be assigned (HTwake, LTwake, SWS, REM)<br/>
* These labels can be propagated to the rest of the recording using the K-Nearest neighbors algorithm<br/>
* Finally, outliers can be detected in the recording using DBSCAN and highlighted in the state dataframe before it gets saved 

****Installation instructions:****
<ol>
<li>Activate the appropriate Anaconda environment (Python version 3.8) </li>
<li>You can install the latest version of setuptools using pip:
<br/>
<code> pip install --upgrade setuptools</code></li>
<li>You can also install build using pip:
<br/>
<code> pip install --upgrade build</code></li>
<li>Navigate to Project directory and run:
<br/>
<code>python -m build </code></li>
<li>You can install dependencies by running:
<br/>
<code>
pip install ./dist/oessa-0.0.1.tar.gz</code></li>
</ol>
