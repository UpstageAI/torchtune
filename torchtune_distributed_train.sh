#!/bin/bash

# ============== SCP 노드 이름 및 IP===================
# upstage-node-001 192.168.1.2
# upstage-node-002 192.168.1.3
# upstage-node-003 192.168.1.4
# upstage-node-004 192.168.1.5
# upstage-node-005 192.168.1.6
# upstage-node-006 192.168.1.7
# upstage-node-007 192.168.1.8
# upstage-node-008 192.168.1.9

#!/bin/bash
BASE_DIR="/app-home/eric/01_workspace/upstage_ai/torchtune"  # FIXME : torchtune directory
CONDA_ENV_NAME="profile_e2e"  # FIXME : 실제 conda 환경 이름으로 변경 필요
CONDA_ROOT="/app-home/eric/anaconda3"  # FIXME : Conda가 설치된 경로

# SSH 접속에 사용할 사용자 이름 (현재 로그인 사용자)
SSH_USER="$(whoami)"

# ============ 설정 변수 ============
NNODES=2 # FIXME : 학습에 사용되는 노드 수
MASTER_NODE="upstage-node-006"   # FIXME : 마스터 노드 이름
MASTER_ADDR="192.168.1.7"        # FIXME : 마스터 노드 내부 IP
MASTER_PORT=29500                 # FIXME : 마스터 노드 포트
NPROC_PER_NODE=8                  # FIXME : 노드당 프로세스 수
NODES=("upstage-node-006" "upstage-node-007")  # FIXME : 학습에 사용되는 GPU 노드 목록

# NCCL 네트워크 인터페이스 설정
NCCL_IFACE="bond-srv.1512"

# 경로 및 환경 설정
CONFIG_PATH="recipes/configs/docev/docev_preview_sample.yaml"  # FIXME : 학습 설정 파일 경로
SCRIPT_PATH="recipes/dev/full_finetune_distributed_ufx_dataset.py"  # FIXME : 학습 스크립트 경로

# 로그 디렉토리 설정
LOG_DIR="${BASE_DIR}/logs"
RUN_NAME="torchtune_distributed_train_test" # FIXME : 학습 실행 이름
LOG_PREFIX="${LOG_DIR}/${RUN_NAME}"

# ============ 실행 준비 ============
echo "===== 분산 학습 시작 준비 ====="
echo "노드 수: $NNODES"
echo "마스터 노드: $MASTER_NODE ($MASTER_ADDR:$MASTER_PORT)"
echo "노드당 프로세스: $NPROC_PER_NODE"
echo "기본 디렉토리: $BASE_DIR"
echo "로그 디렉토리: $LOG_DIR"

# 로그 디렉토리 생성
for NODE in "${NODES[@]}"; do
    ssh -l "$SSH_USER" "$NODE" "mkdir -p $LOG_DIR" || echo "경고: $NODE에서 로그 디렉토리 생성 실패"
done

# ============ 각 노드에서 학습 실행 ============
for ((i=0; i<NNODES; i++)); do
    NODE=${NODES[$i]}
    NODE_RANK=$i
    NODE_LOG_FILE="${LOG_PREFIX}_node${NODE_RANK}.log"

    echo "[$NODE] 노드(랭크 $NODE_RANK)에서 학습 시작"

    ssh -l "$SSH_USER" "$NODE" "bash -l -c '
        # NCCL 환경 변수 설정
        export NCCL_SOCKET_IFNAME=${NCCL_IFACE}

        # Conda 초기화 스크립트 로드
        if [ -f ${CONDA_ROOT}/etc/profile.d/conda.sh ]; then
            source ${CONDA_ROOT}/etc/profile.d/conda.sh
        fi
        # fallback: ~/.bashrc 로드
        source ~/.bashrc 2>/dev/null || true
        conda activate $CONDA_ENV_NAME
        cd $BASE_DIR

        echo \"===== 환경 정보 =====\" > $NODE_LOG_FILE
        echo \"노드: $NODE (랭크 $NODE_RANK)\" >> $NODE_LOG_FILE
        echo \"시작 시간: \$(date)\" >> $NODE_LOG_FILE
        echo \"디렉토리: \$(pwd)\" >> $NODE_LOG_FILE
        echo \"Python: \$(python --version 2>&1)\" >> $NODE_LOG_FILE
        echo \"Conda env: $CONDA_ENV_NAME\" >> $NODE_LOG_FILE
        echo \"===== 학습 시작 =====\" >> $NODE_LOG_FILE

        nohup torchrun \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            --nproc_per_node=$NPROC_PER_NODE \
            $SCRIPT_PATH \
            --config $CONFIG_PATH \
            >> $NODE_LOG_FILE 2>&1 &

        echo \$! > ${LOG_PREFIX}_node${NODE_RANK}.pid
        echo \"PID: \$(cat ${LOG_PREFIX}_node${NODE_RANK}.pid)\" >> $NODE_LOG_FILE
    '"

    sleep 2
    echo "[$NODE] 명령 전송 완료"
done

# 모든 SSH 백그라운드 명령 완료 대기
wait

# 완료 메시지 출력
echo "===== 분산 학습 실행 완료 ====="
echo "로그 파일: ${LOG_PREFIX}_node*.log"
echo "PID 파일: ${LOG_PREFIX}_node*.pid"

echo "-- 중지 명령 --"
for ((i=0; i<NNODES; i++)); do
    NODE=${NODES[$i]}
    echo "ssh -l \"$SSH_USER\" $NODE \"kill \\$(cat ${LOG_PREFIX}_node${i}.pid)\""
done
