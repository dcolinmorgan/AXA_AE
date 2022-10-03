#!/bin/sh

#PBS -l nodes=1:ppn=4
#PBS -l mem=10g
#PBS -l walltime=1:00:00
#PBS -N axa_ae_gcn
#PBS -q stdq1 
###medq1, fatq1, gpuq1, stdq1 cgsd
#PBS -e .GCNerr
#PBS -o .GCNout
#qsub run/AXA_AE_app/axa_ae_gcn_lstm.sh

module load python3 cuda11.0/toolkit/11.0.3 cudnn8.0-cuda11.0/8.0.5.39

module purge
module add anaconda3
module add parallel
# source activate mypy3
source activate mypy38
# source activate myGPU

chmod +x run/AXA_AE_app/axa_ae_gcn_lstm.py
# chmod +x run/AXA_AE_app/axa_ae_gcn_lstm.py

python run/AXA_AE_app/axa_ae_gcn_lstm.py
# python run/AXA_AE_app/axa_ae_gcn_lstm.py


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