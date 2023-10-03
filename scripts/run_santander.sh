for V in 0 1 2 ... 15
do
    python3 scripts/run_santander.py -s $V &
done
