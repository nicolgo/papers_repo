@ECHO off
ECHO "Initial the environment"
CALL conda update conda -y
CALL conda create --name mmgen -y
CALL conda activate mmgen
CALL conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
CALL conda install -c conda-forge opencv -y
CALL pip install -U openmim
CALL mim install mmcv-full
CALL git clone https://github.com/nicolgo/mmgeneration.git
CALL cd mmgeneration
CALL pip3 install -e .
PAUSE
