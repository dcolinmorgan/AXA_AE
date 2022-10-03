#!/bin/sh
# #PBS -N AXA_AE
# #PBS -q gpuq1
# #PBS -l walltime=06:00:00 
# #PBS -l select=1:ncpus=8:ngpus=4:mem=64gb
# #PBS -V
# #PBS -e .AXerr
# #PBS -o .AXout
# #module load python3 cuda11.0/toolkit/11.0.3 cudnn8.0-cuda11.0/8.0.5.39

#PBS -l nodes=1:ppn=24
#PBS -l mem=200g
#PBS -l walltime=96:00:00
#PBS -N axa_ae
#PBS -q stdq1 
###medq1, fatq1, gpuq1
#PBS -e .AXerr
#PBS -o .AXout
#qsub run/AXA_AE/AXA_AE.sh

# module purge
module add anaconda3
# module add parallel
source activate mypy3
# source activate myGPU

chmod +x run/AXA_AE/AXA_AE.py

python run/AXA_AE/AXA_AE.py

# # files=glob.glob('/home/edmondyip/AnE_data/data/*/*')

# filedir='/home/edmondyip/AnE_data/data/AE_attendance_csv/*'
# files=$(ls $filedir*)

# for file in $files
# do
# file=$(eval "echo "$file" | cut -d . -f1")

# # xlsx2csv $file.xlsx -d 'tab' > tmp.txt
# cat $file.csv | tail -n +2 > tmp.txt
# cut -d, -f 1,2,3,4,5,6,7,8,14,41,51,61,66 tmp.txt >> tmp1.txt
# sed 's/\,/\t/g' tmp1.txt > tmp2.txt
# cut -f1,2,4,5,6,7,9,10,11,12,13 tmp2.txt|tr -d '"' >tmp3.txt
# done