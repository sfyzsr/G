rm -rf *.out
sbatch -t 5 -c 4 ./runOMP.sh gauss_OpenMP.c 