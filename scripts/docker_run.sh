CODE_PATH=/home/jihyelee/workspace/nlp
SCRIPT_PATH=${CODE_PATH}/scripts
HF_CACHE=$HOME/huggingface
OUT_PATH=${CODE_PATH}/outputs
DATASET_PATH=${CODE_PATH}/data

sudo docker run -it --gpus '"device='$CUDA_VISIBLE_DEVICES'"' --user=$UID \
  -e HYDRA_FULL_ERROR=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ${HF_CACHE}:/root/.cache/huggingface:rw \
  -v ${CODE_PATH}:/code:ro \
  -v ${SCRIPT_PATH}:/scripts:ro \
  -v ${OUT_PATH}:/outputs:rw \
  -v ${DATASET_PATH}:/dataset:rw \
ckl:v4.22.2 $@