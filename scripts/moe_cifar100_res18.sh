
dataset=cifar100
net=res18-moe2 # vgg16-moe/wrn28-moe

ft_attack(){
    CUDA_VISIBLE_DEVICES=$4 python attack_moe_ft.py --dataset ${dataset} --net $net \
            --n-expert 4 --ratio 1.0 \
            --num_shadow $5 --num_query 1000 --num_augment $1 --shadow_dir $2 \
            --seed $3
}

# ============================== a trial =============================
group(){

nepoch=100
CUDA_VISIBLE_DEVICES=0 python train_moe.py --dataset ${dataset} --net $net \
    --bs 64 --lr 0.1 --pkeep 0.8 --n-expert 4 --num_shadow 128 --pathway_num $4 \
    --shadow_id 0 --n_epochs ${nepoch} --dis_loss $1 --alpha $2 --beta $3 & 

CUDA_VISIBLE_DEVICES=1 python train_moe.py --dataset ${dataset} --net $net \
    --bs 64 --lr 0.1 --pkeep 0.8 --n-expert 4 --num_shadow 128 --pathway_num $4 \
    --shadow_id 2 --n_epochs ${nepoch} --dis_loss $1 --alpha $2 --beta $3 & 
    
CUDA_VISIBLE_DEVICES=2 python train_moe.py --dataset ${dataset} --net $net \
    --bs 64 --lr 0.1 --pkeep 0.8 --n-expert 4 --num_shadow 128 --pathway_num $4 \
    --shadow_id 4 --n_epochs ${nepoch} --dis_loss $1 --alpha $2 --beta $3 & 

CUDA_VISIBLE_DEVICES=3 python train_moe.py --dataset ${dataset} --net $net \
    --bs 64 --lr 0.1 --pkeep 0.8 --n-expert 4 --num_shadow 128 --pathway_num $4 \
    --shadow_id 6 --n_epochs ${nepoch} --dis_loss $1 --alpha $2 --beta $3 & 

wait

pre_trained_moe="${dataset}-$net-0.8-$nepoch-$1-$2-$3"
echo $pre_trained_moe

ft_ep=10
CUDA_VISIBLE_DEVICES=0 python train_moe_ft.py --dataset ${dataset} --net $net \
         --pre_trained $pre_trained_moe \
         --bs 64 --lr 0.1 --pkeep 0.8 --n-expert 4 --shadow_id 0 --save_shadow_id 1 \
         --num_shadow 128 --n_epochs ${ft_ep} --pathway_num $4 \
         --member_num 5000 &

CUDA_VISIBLE_DEVICES=1 python train_moe_ft.py --dataset ${dataset} --net $net \
         --pre_trained $pre_trained_moe \
         --bs 64 --lr 0.1 --pkeep 0.8 --n-expert 4 --shadow_id 2 --save_shadow_id 3 \
         --num_shadow 128 --n_epochs ${ft_ep} --pathway_num $4 \
         --member_num 5000 &

CUDA_VISIBLE_DEVICES=2 python train_moe_ft.py --dataset ${dataset} --net $net \
         --pre_trained $pre_trained_moe \
         --bs 64 --lr 0.1 --pkeep 0.8 --n-expert 4 --shadow_id 4 --save_shadow_id 5 \
         --num_shadow 128 --n_epochs ${ft_ep} --pathway_num $4 \
         --member_num 5000 &

CUDA_VISIBLE_DEVICES=3 python train_moe_ft.py --dataset ${dataset} --net $net \
         --pre_trained $pre_trained_moe \
         --bs 64 --lr 0.1 --pkeep 0.8 --n-expert 4 --shadow_id 6 --save_shadow_id 7 \
         --num_shadow 128 --n_epochs ${ft_ep} --pathway_num $4 \
         --member_num 5000 &

wait 

cp saved_models/${pre_trained_moe}/$net_shadow_* saved_models/ft-${pre_trained_moe}-e${ft_ep}/
dir=ft-${pre_trained_moe}-e${ft_ep}

for i in {1..10}
do
    ft_attack 64 $dir $((i * 3)) 0 2 
    ft_attack 64 $dir $((i * 3)) 0 8
    wait
done
}

group no 0.0 0.0 64 # offline lira
group orth 0.0 0.01 64 # online lira

