for V in $(seq 0 15);
do
    python3 scripts/run_gbmep_full.py -s $V &
done