#!/bin/bash
#SBATCH --job-name=c100-baseline
#SBATCH --account=IscrC_SIMP
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun/logs/baseline-%j.out
#SBATCH --error=/leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun/logs/baseline-%j.err
set -euo pipefail
if [[ "${SLURM_JOB_ACCOUNT:-}" != "iscrc_simp" && "${SLURM_JOB_ACCOUNT:-}" != "IscrC_SIMP" ]]; then
  echo "Refusing to run outside IscrC_SIMP." >&2
  exit 2
fi
cd /leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun
source env_setup.sh
echo "==> $(date) job=${SLURM_JOB_ID:-N/A} node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python prepare_cifar100_hf.py
TARGET=${TARGET:-0.70}
C100_RUNS=${RUNS:-30} C100_EPOCHS=${EPOCHS:-16} C100_SLEEP_CYCLES=1000000000 python train_cifar100_resnet_muon.py
python analyze_cifar100.py "/leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun/logs/baseline-${SLURM_JOB_ID}.out" --target "$TARGET" || true
echo "==> done $(date)"
