#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-30:00     # DD-HH:MM:SS

module load python/3.6

python -V

module load cuda/10.0

module load nixpkgs/16.09

module load gcc/7.3.0

nvcc -V

module spider cudnn/7.6

SOURCEDIR=~/projects/def-gnico/kacemkh/knockoffnets-old


#Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/knockoffnets-env
source $SLURM_TMPDIR/knockoffnets-env/bin/activate
pip install --no-index -r  $SOURCEDIR/requirements.txt

pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

#install pretrained models
#git clone https://github.com/Cadene/pretrained-models.pytorch.git
#cd pretrained-models.pytorch
#python setup.py install


# Prepare data
mkdir $SLURM_TMPDIR/ILSVRC2012
#tar -xvf ~/scratch/kacem/datasets/ILSVRC2012/ILSVRC2012_img_train.tar -C $SLURM_TMPDIR/ILSVRC2012
tar -xvf ~/scratch/kacem/datasets/ILSVRC2012/ILSVRC2012_img_train_t3.tar -C $SLURM_TMPDIR/ILSVRC2012

cd $SLURM_TMPDIR/ILSVRC2012
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

#mkdir $SLURM_TMPDIR/data
#tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

# Start training
# python $SOURCEDIR/train.py $SLURM_TMPDIR/data
cd $SOURCEDIR

python  -m "knockoff.victim.train"  CIFAR100 densenet -d 0 \
        -o $SOURCEDIR/models/victim/cifar100-densenet -e 1 --log-interval 25
printf  "\n\nend of victim.train\n\n"

python -m "knockoff.adversary.transfer" random $SOURCEDIR/models/victim/cifar100-densenet \
        --out_dir $SOURCEDIR/models/adversary/cifar100-densenet-random --budget 1000 \
        --queryset ImageNet1k $SLURM_TMPDIR/ILSVRC2012 --batch_size 10 -d 0

printf  "\n\nend of adversary.transfer\n\n"

python -m "knockoff.adversary.train" $SOURCEDIR/models/adversary/cifar100-densenet-random \
        resnet34 CIFAR100 --budgets 1000 -d 0 --pretrained imagenet \
        --log-interval 100 --epochs 20 --lr 0.01

printf  "\n\nend of adversary.train\n\n"
#launch the script with: sbatch --gres=gpu:1 --cpus-per-task=6 --mem=32000M --time=0-00:10 train-script.sh