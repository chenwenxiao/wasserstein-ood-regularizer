for i in {1..10}; do (bash dask_run.sh -- bash pretrain_start.sh ${1} $i); done