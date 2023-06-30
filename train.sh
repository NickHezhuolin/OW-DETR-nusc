export CUDA_VISIBLE_DEVICES=1

set -x

EXP_DIR=exps/OWDETR_t1
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod_nusc --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 8 --data_root '/home/hez4sgh/1_workspace/OW-DETR-nusc/data/OWDETR' --train_set 't1_train_new_split' --test_set 't1_test_new_split' --num_classes 11 \
    --unmatched_boxes --epochs 45 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}