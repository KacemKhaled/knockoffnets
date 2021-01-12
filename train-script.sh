#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-30:00     # DD-HH:MM:SS

module load python/3.6 cuda cudnn

SOURCEDIR=~/projects/def-gnico/kacemkh/knockoffnets-old


#Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/knockoffnets-env
source $SLURM_TMPDIR/knockoffnets-env/bin/activate
pip install --no-index -r  $SOURCEDIR/requirements.txt

#install pretrained models
git clone https://github.com/Cadene/pretrained-models.pytorch.git
cd pretrained-models.pytorch
python setup.py install
cd ..

#pip install --no-index -r  $SOURCEDIR/requirements.txt


# Prepare data
mkdir $SLURM_TMPDIR/ILSVRC2012
tar -xvf ~/scratch/kacem/datasets/ILSVRC2012/ILSVRC2012_img_train.tar -C $SLURM_TMPDIR/ILSVRC2012

cd $SLURM_TMPDIR/ILSVRC2012
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

#mkdir $SLURM_TMPDIR/data
#tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

# Start training
# python $SOURCEDIR/train.py $SLURM_TMPDIR/data
cd $SOURCEDIR

python  -m "knockoff.victim.train"  FashionMNIST lenet -d 0 \
        -o $SOURCEDIR/models/victim/fashionmnist-lenet -e 10 --log-interval 25

python -m "knockoff.adversary.transfer" random $SOURCEDIR/models/victim/fashionmnist-lenet \\
        --out_dir $SOURCEDIR/models/adversary/fashionmnist-lenet-random --budget 5000 \\
        --queryset ImageNet1k --batch_size 8 -d 0

#launch the script with: sbatch --gres=gpu:1 --cpus-per-task=6 --mem=32000M --time=0-00:30 train-script.sh