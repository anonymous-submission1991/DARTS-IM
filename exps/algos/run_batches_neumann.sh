epo=0
while [ $epo -le 2 ]
  do 
     CUDA_VISIBLE_DEVICES=0 python3 run_example_batches_neumann.py \
--batches=1 --rand_seed=$epo
     ((epo++))
 done
 
 
epo=0
while [ $epo -le 2 ]
  do 
     CUDA_VISIBLE_DEVICES=0 python3 run_example_batches_neumann.py \
--batches=10 --rand_seed=$epo
     ((epo++))
 done
 
epo=0
while [ $epo -le 2 ]
  do 
     CUDA_VISIBLE_DEVICES=0 python3 run_example_batches_neumann.py \
--batches=20 --rand_seed=$epo
     ((epo++))
 done
 
epo=0
while [ $epo -le 2 ]
  do 
     CUDA_VISIBLE_DEVICES=0 python3 run_example_batches_neumann.py \
--batches=30 --rand_seed=$epo
     ((epo++))
 done
 
 
 
epo=0
while [ $epo -le 2 ]
  do 
     CUDA_VISIBLE_DEVICES=0 python3 run_example_batches_neumann.py \
--batches=40 --rand_seed=$epo
     ((epo++))
 done
 
epo=0
while [ $epo -le 2 ]
  do 
     CUDA_VISIBLE_DEVICES=0 python3 run_example_batches_neumann.py \
--batches=50 --rand_seed=$epo
     ((epo++))
 done
