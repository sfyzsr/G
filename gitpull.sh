git fetch --all
git reset --hard origin/main
git fetch
rm -rf *.out
sbatch -t 5 -c 4 ./sb.sh g.c 
