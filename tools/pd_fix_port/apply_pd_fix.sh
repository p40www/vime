#!/bin/bash
# 476 PD 挂死修复移植（参考 s10 27258109）
# 用法：在容器内执行  bash /workspace/start_run/apply_pd_fix.sh
set -u
VLLM=/vllm-workspace/vllm
ASCEND=/vllm-workspace/vllm-ascend
PATCH=/workspace/start_run/port-27258109-mooncake.patch
say(){ echo -e "\n=== $* ==="; }

say "0) 预检"
test -d "$VLLM/vllm" && test -d "$ASCEND/vllm_ascend" || { echo "ERROR: 源码树不存在"; exit 1; }
(cd "$ASCEND" && git status --porcelain | head -5)

cat > "$PATCH" <<'PATCH_EOF'
diff --git a/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py b/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py
index 181c6b2..65b622a 100644
--- a/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py
+++ b/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py
@@ -338,6 +338,15 @@ class MooncakeBaseConnectorWorker:
             register_regions.lengths,
         )
 
+    def prepare_sleep(self) -> None:
+        # EngineCore has paused/drained the scheduler. Finish local device work
+        # before dropping the registrations which pin CaMem's physical pages.
+        torch.npu.synchronize()
+        global_te.unregister_buffer()
+
+    def resume_after_wake(self) -> None:
+        global_te.reregister_buffer()
+
     def get_finished(self) -> tuple[set[str], set[str]]:
         """Compatibility wrapper for the legacy completion API."""
         results = self.get_transfer_results()
diff --git a/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py b/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py
index 5486bb7..8d256d8 100644
--- a/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py
+++ b/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py
@@ -171,6 +171,14 @@ class MooncakeBaseConnector(KVConnectorBase_V1, SupportsHMA):
         assert self.connector_worker is not None
         self.connector_worker.register_kv_caches(kv_caches)
 
+    def prepare_sleep(self) -> None:
+        assert self.connector_worker is not None
+        self.connector_worker.prepare_sleep()
+
+    def resume_after_wake(self) -> None:
+        assert self.connector_worker is not None
+        self.connector_worker.resume_after_wake()
+
     def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
         """Return requests whose receive and send operations have finished."""
         assert self.connector_worker is not None
diff --git a/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py b/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py
index 9693200..b4d9a53 100644
--- a/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py
+++ b/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py
@@ -47,6 +47,7 @@ from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
     group_concurrent_contiguous,
     zmq_ctx,
 )
+from vllm_ascend.distributed.kv_transfer.utils.utils import get_transfer_timeout_value
 
 
 class MooncakePullRecvingThread(threading.Thread):
@@ -1576,6 +1577,18 @@ class MooncakePullConnectorWorker(MooncakeBaseConnectorWorker):
             if not self._recving_thread.is_alive():
                 raise RuntimeError("Mooncake pull receiving thread failed to start")
 
+    def prepare_sleep(self) -> None:
+        if self._recving_thread is not None:
+            # A scheduler pause can abort requests while their remote READ is
+            # still in flight. NPU stream synchronization alone cannot fence it.
+            pending = self._recving_thread.request_queue
+            with pending.all_tasks_done:
+                if not pending.all_tasks_done.wait_for(
+                    lambda: pending.unfinished_tasks == 0, timeout=min(get_transfer_timeout_value(), 300)
+                ):
+                    raise TimeoutError("Mooncake pulls did not drain before sleep")
+        super().prepare_sleep()
+
     def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
         """Start the pull operations described by scheduler metadata."""
         if not metadata.requests:
diff --git a/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py b/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py
index 7cf309b..9dfce8d 100644
--- a/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py
+++ b/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py
@@ -1,43 +1,64 @@
 import threading
 
 
 class GlobalTE:
     def __init__(self):
         self.transfer_engine = None
         self.is_register_buffer: bool = False
+        self._registered_buffers: dict[int, int] = {}
+        self._active_buffers: set[int] = set()
         self.transfer_engine_lock = threading.Lock()
         self.register_buffer_lock = threading.Lock()
 
     def get_transfer_engine(self, hostname: str, device_name: str | None):
         if self.transfer_engine is None:
             with self.transfer_engine_lock:
                 # Double-Checked Locking
                 if self.transfer_engine is None:
                     try:
                         from mooncake.engine import TransferEngine  # type: ignore
                     except ImportError as e:
                         raise ImportError(
                             "Please install mooncake by following the instructions at "
                             "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md"  # noqa: E501
                             "to run vLLM with MooncakeConnector."
                         ) from e
                     self.transfer_engine = TransferEngine()
                     device_name = device_name if device_name is not None else ""
                     ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                     if ret_value != 0:
                         raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
         return self.transfer_engine
 
     def register_buffer(self, ptrs: list[int], sizes: list[int]):
         with self.register_buffer_lock:
             assert self.transfer_engine is not None, "Transfer engine must be initialized"
-            if self.is_register_buffer:
-                return
-            for ptr, size in zip(ptrs, sizes):
+            for ptr, size in zip(ptrs, sizes, strict=True):
+                if ptr in self._active_buffers:
+                    if self._registered_buffers[ptr] != size:
+                        raise ValueError("Cannot resize an active Mooncake memory registration")
+                    continue
                 ret_value = self.transfer_engine.register_memory(ptr, size)
                 if ret_value != 0:
-                    raise RuntimeError("Mooncake memory registration failed.")
-            self.is_register_buffer = True
+                    raise RuntimeError(f"Mooncake memory registration failed: ptr={ptr}, size={size}, ret={ret_value}")
+                self._registered_buffers[ptr] = size
+                self._active_buffers.add(ptr)
+            self.is_register_buffer = bool(self._active_buffers)
+
+    def unregister_buffer(self):
+        """Release device registrations before CaMem unmaps physical pages."""
+        with self.register_buffer_lock:
+            for ptr in list(self._active_buffers):
+                ret_value = self.transfer_engine.unregister_memory(ptr)
+                if ret_value != 0:
+                    raise RuntimeError(f"Mooncake memory deregistration failed: ptr={ptr}, ret={ret_value}")
+                self._active_buffers.remove(ptr)
+            self.is_register_buffer = False
+
+    def reregister_buffer(self):
+        # CaMem restores the same virtual addresses, but the physical pages and
+        # their driver registrations are new. Weight-only wakes must not call us.
+        self.register_buffer(list(self._registered_buffers), list(self._registered_buffers.values()))
 
 
 global_te = GlobalTE()
PATCH_EOF

say "1) Mooncake 四文件（B 段）"
cd "$ASCEND"
if git apply --check "$PATCH" 2>/dev/null; then
  git apply "$PATCH" && echo "APPLIED (mooncake)"
elif git apply --check -C1 "$PATCH" 2>/dev/null; then
  git apply -C1 "$PATCH" && echo "APPLIED (-C1, mooncake)"
else
  echo "WARN: mooncake 补丁上下文不匹配，跳过（不影响非 colocate 的挂死验证；colocate 时需手工移植）"
fi
grep -q "global_te" "$ASCEND/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py" \
  && echo "base_worker global_te OK" || echo "WARN: base_worker 里没有 global_te（colocate 路径需要）"

say "2) vLLM 调度器空 output 兜底（A 段，治挂死的关键）"
python3 - <<'PY'
from pathlib import Path
p = Path("/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py")
s = p.read_text()
if "A no-forward worker step" in s:
    print("already patched"); raise SystemExit(0)
lines = s.splitlines(keepends=True)
target = "self._update_from_kv_xfer_finished(kv_connector_output)"
guard = [
    "        if self.connector is not None and kv_connector_output is None:\n",
    "            # A no-forward worker step may discard its empty connector output.\n",
    "            # Scheduler-side connectors still need to consume remote ACKs (for\n",
    "            # example Mooncake producer delayed frees), including while paused.\n",
    "            kv_connector_output = KVConnectorOutput()\n",
]
out, done = [], False
for i, line in enumerate(lines):
    if (not done and line.strip() == "if kv_connector_output:"
            and i + 1 < len(lines) and target in lines[i + 1]):
        out.extend(guard); done = True
    out.append(line)
if not done:
    print("ANCHOR NOT FOUND: 需要手工定位 `if kv_connector_output:`")
    for i, l in enumerate(lines):
        if target in l: print("  line", i + 1, l.rstrip())
    raise SystemExit(2)
p.write_text("".join(out))
print("scheduler patched")
PY

say "3) vllm-ascend worker.py（C 段）"
python3 - <<'PY'
from pathlib import Path
p = Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py")
s = p.read_text()
if "_prepare_kv_transfer_sleep" in s:
    print("already patched"); raise SystemExit(0)

if "has_kv_transfer_group" not in s:
    import subprocess
    ok = subprocess.run(
        ["python3", "-c", "from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group"],
        capture_output=True).returncode == 0
    if not ok:
        print("ANCHOR: 导入路径不可用，请手工确认 has_kv_transfer_group 的导入方式"); raise SystemExit(2)
    lines = s.splitlines(keepends=True)
    idx = next(i for i, l in enumerate(lines) if l.startswith("import gc"))
    lines.insert(idx + 1, "from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group\n")
    s = "".join(lines)

lines = s.splitlines(keepends=True)
methods = [
    "    def _prepare_kv_transfer_sleep(self) -> None:\n",
    "        if has_kv_transfer_group():\n",
    "            prepare_sleep = getattr(get_kv_transfer_group(), \"prepare_sleep\", None)\n",
    "            if prepare_sleep is not None:\n",
    "                prepare_sleep()\n",
    "\n",
    "    def _resume_kv_transfer_after_wake(self, tags: list[str] | None) -> None:\n",
    "        if has_kv_transfer_group() and (tags is None or \"kv_cache\" in tags):\n",
    "            resume_after_wake = getattr(get_kv_transfer_group(), \"resume_after_wake\", None)\n",
    "            if resume_after_wake is not None:\n",
    "                resume_after_wake()\n",
    "\n",
]
idx = next((i for i, l in enumerate(lines) if l.strip().startswith("def sleep(self")), None)
if idx is None:
    print("ANCHOR NOT FOUND: def sleep"); raise SystemExit(2)
lines[idx:idx] = methods
s = "".join(lines)

lines = s.splitlines(keepends=True)
idx = next(i for i, l in enumerate(lines) if l.strip().startswith("def sleep(self"))
call_idx = next((i for i in range(idx, len(lines)) if lines[i].strip() == "rl_config = get_ascend_config().rl_config"), None)
if call_idx is None:
    print("ANCHOR NOT FOUND: sleep 内 rl_config"); raise SystemExit(2)
lines.insert(call_idx, "        self._prepare_kv_transfer_sleep()\n")

wake_idx = next((i for i, l in enumerate(lines) if l.strip() == "self.sleep_wakeup_manager.wakeup(tags)"), None)
if wake_idx is None:
    print("ANCHOR NOT FOUND: wakeup(tags)"); raise SystemExit(2)
lines.insert(wake_idx + 1, "        self._resume_kv_transfer_after_wake(tags)\n")
Path("/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py").write_text("".join(lines))
print("worker patched")
PY

say "4) 语法检查"
python3 -m py_compile "$VLLM/vllm/v1/core/sched/scheduler.py" && echo "scheduler py_compile OK"
python3 -m py_compile "$ASCEND/vllm_ascend/worker/worker.py" && echo "worker py_compile OK"

echo -e "\n完成。跑验证：bash /workspace/start_run/verify_pd_fix.sh"
