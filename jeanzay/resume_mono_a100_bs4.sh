#!/bin/bash
#SBATCH --job-name=turbo_pix2pix_a100          # nom du job
# Il est possible d'utiliser une autre partition que celle par défaut
# en activant l'une des 5 directives suivantes :
##SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
#SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
# Ici, reservation de 10 CPU (pour 1 tache) et d'un GPU sur un seul noeud :
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU (soit 1/4 ou 1/8 des GPU du noeud suivant la partition),
# l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour la seule tache:
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 des CPU du noeud 8-GPU)
##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 des CPU du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=jz/turbo_pix2pix_a100_%j.out      # nom du fichier de sortie
#SBATCH --error=jz/turbo_pix2pix_a100_%j.err       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --account=abj@a100
#SBATCH --mail-user=nicolas.gonthier@ign.fr
#SBATCH --mail-type=END,FAIL

# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition
module load cpuarch/amd
 
# Chargement des modules
module load pytorch-gpu/py3/2.0.1
 
# Echo des commandes lancees
set -x
 
export HF_HOME='/lustre/fsn1/projects/rech/abj/ujq24es/huggingface'
export WANDB_MODE="offline"
# Pour la partition "gpu_p5", le code doit etre compile avec les modules compatibles
# Execution du code
#--report_to "tensorboard"
#    --track_val_fid \
accelerate config
accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="/lustre/fswork/projects/rech/abj/ujq24es/img2img-turbo/output/pix2pix_turbo/flair_bs4_r5/checkpoints/model_14001.pkl" \
    --output_dir="output/pix2pix_turbo/flair_bs4_r6" \
    --dataset_folder="/lustre/fsn1/projects/rech/abj/ujq24es/dataset/PixtoPixTurbo_FLAIR" \
    --resolution=512 \
    --train_batch_size=4 \
    --enable_xformers_memory_efficient_attention --viz_freq 5000 \
    --num_samples_eval 10 \
    --num_training_epochs 2 --max_train_steps 10000 \
    --dataloader_num_workers 8 \
    --checkpointing_steps 2000 \
    --report_to "wandb" --tracker_project_name "pix2pix_turbo_flair_a100_bs4_r3"

## "/lustre/fswork/projects/rech/abj/ujq24es/img2img-turbo/output/pix2pix_turbo/flair_bs4/checkpoints/model_14001.pkl" 
## "/lustre/fswork/projects/rech/abj/ujq24es/img2img-turbo/output/pix2pix_turbo/flair_bs4_r1/checkpoints/model_14001.pkl" # 28000 iter
## "/lustre/fswork/projects/rech/abj/ujq24es/img2img-turbo/output/pix2pix_turbo/flair_bs4_r2/checkpoints/model_14001.pkl" # 42000 iter
## "/lustre/fswork/projects/rech/abj/ujq24es/img2img-turbo/output/pix2pix_turbo/flair_bs4_r3/checkpoints/model_14001.pkl" # 56000 iter
## "/lustre/fswork/projects/rech/abj/ujq24es/img2img-turbo/output/pix2pix_turbo/flair_bs4_r4/checkpoints/model_14001.pkl" # 70000 iter
## "/lustre/fswork/projects/rech/abj/ujq24es/img2img-turbo/output/pix2pix_turbo/flair_bs4_r5/checkpoints/model_14001.pkl" # 84000 iter