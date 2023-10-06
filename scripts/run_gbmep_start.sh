for V in $(seq 0 15);
do
    python3 scripts/run_gbmep_start.py -s $V &
done
