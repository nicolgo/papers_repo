@ECHO off
ECHO "Initial the environment"
CALL conda update conda -y
CALL conda create --name pointflow -y
CALL conda activate pointflow
CALL conda install -c open3d-admin open3d -y
CALL conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
CALL conda install -c conda-forge torchdiffeq tensorboardx -y
CALL conda install matplotlib tensorboard numpy scipy scikit-learn -y
PAUSE
