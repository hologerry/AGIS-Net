set -ex
# misc
GPU_ID=$1   # gpu id

# models
RESULTS_DIR='./results/base_gray_color'
MODEL='agisnet'

# dataset
CLASS='base_gray_color'
DATASET_MODE='multi_fusion'
PHASE='val'

NUM_TEST=260000

DIRECTION='AtoC'
LOAD_SIZE=64
FINE_SIZE=64

INPUT_NC=3
NENCODE=4
FEW_SIZE=0

RESIZE_OR_CROP='none'
NO_FLIP='--no_flip'

NEF=32
NGF=32
NDF=32

NET_G='agisnet'
NET_D='basic_64'
NET_D2='basic_64'
NET_DLOCAL='basic_32'

USE_ATTENTION='--use_attention'
CONDITIONAL_D='--conditional_D'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./pretrained_models/ \
  --nencode ${NENCODE} \
  --name ${CLASS} \
  --phase ${PHASE} \
  --direction ${DIRECTION} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --resize_or_crop ${RESIZE_OR_CROP} \
  --input_nc ${INPUT_NC} \
  --model ${MODEL} \
  --ngf ${NGF} \
  --ndf ${NDF} \
  --nef ${NEF} \
  --netG ${NET_G} \
  --netD ${NET_D} \
  --netD_B ${NET_D2} \
  --netD_local ${NET_DLOCAL} \
  --use_dropout \
  ${USE_ATTENTION} \
  --dataset_mode ${DATASET_MODE} \
  --num_test ${NUM_TEST} \
  --no_flip