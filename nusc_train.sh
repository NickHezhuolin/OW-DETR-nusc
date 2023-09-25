export CUDA_VISIBLE_DEVICES=1

set -x

EXP_DIR=exps/OWDETR_nusc_t1
PY_ARGS=${@:1}

# owdetr-nusc-train
python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod_nusc --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 5 --data_root 'data/OWDETR' --train_set 't1_nusc_5cls_train_split' --test_set 't1_nusc_5cls_val_split' --num_classes 11 \
    --unmatched_boxes --epochs 45 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}