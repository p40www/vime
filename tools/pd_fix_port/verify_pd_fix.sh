#!/bin/bash
# 验证 PD 挂死修复：完整跑 0.6B 用例
# 用法：在容器内执行  bash /workspace/start_run/verify_pd_fix.sh
bash /workspace/start_run/clean.sh
cd /root/vime
export PYTHONPATH="/root/Megatron-LM:/root/vime:/root/MegatronAdaptor:/root/TransformerEngineNPU:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge"
export VIME_PLATFORM=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=enp67s0f5
export GLOO_SOCKET_IFNAME=enp67s0f5
export TP_SOCKET_IFNAME=enp67s0f5
export HCCL_IF_IP=192.168.0.164
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/workspace
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_ASCEND_ENABLE_NZ=0
export VIME_TEST_DISPOSE_ON_FAILURE=1

mkdir -p /workspace/logs/pd
python -u tests/test_qwen3_0.6B_pd_npu.py > /workspace/logs/pd/port-verify.log 2>&1
echo "EXIT=$?"
echo "日志: /workspace/logs/pd/port-verify.log"
echo "--- 关键判据 ---"
grep -n "Timer update_weights end\|Timer update_weights start\|PD consistency comparison\|Traceback\|Free memory" /workspace/logs/pd/port-verify.log | tail -20
