@ECHO off
ECHO "Initial the environment"
CALL conda create --name ae_gen python=3.9 -y
CALL conda activate ae_gen
CALL conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
CALL conda install h5py tqdm tensorboard numpy scipy scikit-learn -y
PAUSE
