rm -rf *.out
sbatch -t 5 -n 5 ./runMpi.sh gauss_MPI.c 