rm -rf *.out
sbatch -t 5 -n 4 ./runMpi.sh gauss_MPI.c 