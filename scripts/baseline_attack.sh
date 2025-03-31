
dataset=cifar100 # cifar10
net=res18-ori # vgg16-ori/wrn28

baseline(){
    CUDA_VISIBLE_DEVICES=0 python attack.py --dataset $dataset --name $net --net $net \
        --shadow_dir $1 --target_dir $1 --num_shadow $2 \
        --num_query 1000 --trial 10
}

dir=$net-$dataset-random-0.5
baseline $dir 4
for i in {1..16}
do
    baseline $dir $((i * 8))
done