@ECHO off
ECHO "Initial the environment"
CALL conda update conda -y
CALL conda create --name common_env -y
CALL conda activate common_env
CALL conda install -c open3d-admin open3d -y
CALL conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
CALL conda install -c conda-forge torchdiffeq tensorboardx -y
CALL conda install matplotlib h5py tqdm tensorboard numpy scipy scikit-learn -y
PAUSE
