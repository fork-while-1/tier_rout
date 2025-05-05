#!/bin/bash
sbatch --requeue <<EOT
#!/bin/bash
#SBATCH -J $1
#SBATCH -o output/stdout/$1.out
#SBATCH -e output/stderr/$1.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --get-user-env
#SBATCH --mem 300G
#SBATCH -t 99:00:00
#SBATCH --partition=default_partition
source /home/tt544/.bashrc
conda activate mlifc
export TMPDIR=/share/suh-scrap2/tmp
/home/tt544/tier_rout/run_clm.py `/home/tt544/tier_rout/config.py /home/tt544/tier_rout/configs/16/$1.json`
EOT