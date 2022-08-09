@ECHO off
ECHO "Initial the environment"
CALL conda update conda -y
CALL conda create --name ddpm python=3.8 -y
CALL conda activate ddpm
CALL conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
CALL conda install -c conda-forge einops -y
CALL conda install tqdm pillow -y
CALL conda install -c fastai accelerate -y
CALL pip install ema-pytorch
PAUSE
