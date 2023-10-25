for V in $(seq 0 15);
do
    python3 scripts/scripts_2020/run_gbmep_full_2020.py -s $V &
done