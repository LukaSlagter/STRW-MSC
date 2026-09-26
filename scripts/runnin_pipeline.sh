#!/bin/bash

# If you get a permission denied error for run.sh itself, run this line in the terminal:
# chmod +x ./run.sh

# If you get errors about weird (return) characters, and/or you edited run.sh on Windows, run this command:
# dos2unix ./run.sh &>/dev/null

# Make sure you do NOT run in a virtual environment (e.g. conda, uv), or your results may be different than when we run your code
# On the STRW computers, you may need to run "module purge" if you load any modules at startup

matplotlibversion="$( python3 -m pip list | grep "matplotlib " | tr -s ' ' | cut -d' ' -f2 )"
if [ "${matplotlibversion}" != "3.9.0" ] ; then
    echo "WARNING: Matplotlib version is different from the default vdesk one (${matplotlibversion} vs 3.9.0), this may or may not cause differences/errors."
fi
numpyversion="$( python3 -m pip list | grep "numpy " | tr -s ' ' | cut -d' ' -f2 )"
if [ "${numpyversion}" != "1.26.4" ] ; then
    echo "WARNING: Numpy version is different from the default vdesk one (${numpyversion} vs 1.26.4), this may or may not cause differences/errors."
fi

# Check if black formatter is installed
if ! python3 -m black --version &>/dev/null ; then
    echo "Black formatter not found. Installing..."
    python3 -m pip install black
fi
# Format all python files (note that this assumes your python files are all in the same directory as run.sh)
echo "Uniformly formatting Python code..."
python3 -m black .


echo "Running Python Pipeline..."
python3 Running_pipeline.py


echo "!!!! All done !!!"
