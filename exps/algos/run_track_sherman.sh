epo=0
while [ $epo -le 2 ]
  do 
     CUDA_VISIBLE_DEVICES=0 python3 run_track_sherman.py \
--batches=30 --rand_seed=$epo
     ((epo++))
 done
 
