epo=0
while [ $epo -le 2 ]
  do 
     CUDA_VISIBLE_DEVICES=0 python3 run_track_neumann.py \
--batches=1 --rand_seed=$epo
     ((epo++))
 done
 
