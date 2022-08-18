@ECHO off
ECHO "Initial the environment"
CALL conda update conda -y
CALL conda create --name iddpm -y
CALL conda activate iddpm
CALL conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
CALL conda install tqdm tensorboard numpy scipy scikit-learn -y
CALL conda install -c conda-forge mpi4py
CALL pip install blobfile
PAUSE
