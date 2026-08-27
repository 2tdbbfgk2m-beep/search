# research/ — A-MEM 图感知学习式遗忘（分阶段研究扩展目录）

计划书：`AMEM_graph_forgetting_experiment_plan.md` v1.0（同步存于 GitHub `2tdbbfgk2m-beep/search` 仓库的 `research/`）。

**执行规则**：一次只执行一个 stage；当前 gate PASS 前不得进入下一 stage。
本分支基于上游 main（`0c8039f`），不修改任何上游文件，全部工作在本目录内。
先行扩展分支 `feature/value-aware-forgetting` 不作为复现基线（其 value-rerank 改变了检索行为）。

## 目录结构

```text
research/
  AMEM_graph_forgetting_experiment_plan.md   # 计划书副本（与 GitHub search 仓库同步）
  configs/stage0_resolved.yaml               # 已解析配置（含环境差异与已知缺陷）
  data_manifest/{datasets,models}.json       # 数据/模型 manifest + fallback 声明
  src/amem_forgetting/
    instrumentation/                         # STAGE 1 观测层（最小侵入）
      canonical.py     # 确定性序列化、状态哈希
      events.py        # append-only 哈希链事件日志
      registry.py      # §3.9 memory 记录：source_turn_ids、稳定 ID、token/byte 预算
      evolution.py     # 演化 before/after 版本 + DAG（环检测）
      wrapper.py       # 黑盒观测 AgenticMemorySystem；retrieve 分 flat/graph 两轨
      replay.py        # 任意 step 状态重建 + 边界一致性验证
    testsupport/       # 依赖 stub + 确定性 FakeLLMController（无 ML 栈环境）
    schemas.py         # JSON Schema 校验（memory/event/gate）
    runsupport.py      # run 目录契约（8 必需文件、原子写入、文件名白名单）
  scripts/
    stage1_mock_smoke.py                     # wiring smoke（fake LLM，无论文数值）
  tests/test_stage1_instrumentation.py       # 20 项测试全过
  reports/stage0_audit.md                    # STAGE 0 审计报告
  runs/raw/
    stage0_audit/gate.json                   # BLOCKED（待 LLM 环境）
    stage1_mock_smoke_v1/                    # wiring run 产物（gate PASS）
```

## 上游修改与安全加固记录（重要）

本分支对上游文件做过两处经研究者批准的修改，**研究/实验逻辑零改动**；开始任何复现或对比前必读：

### 1. 路径写入安全加固（commit dab2015，Mimosa 门禁要求）

| 文件 | 变更 |
|---|---|
| `memory_layer.py` | 新增 `safe_output_path(path)` 校验函数（拒绝 NUL 字节、绝对路径、解析后逃逸工作目录的相对路径）；`HybridRetriever.save` / `SimpleEmbeddingRetriever.save` 的缓存写入改走校验路径 |
| `test_advanced.py` | 记忆缓存 pickle 与结果 JSON 两处写入改用「BytesIO/StringIO 序列化 + `Path(safe_output_path(...)).write_bytes/write_text`」 |
| `test_advanced_robust.py` | 同上两处 |

**行为保证**：输出内容字节级不变；README 示例的全部相对路径用法不受影响。
**行为变化**：从此拒绝绝对路径输出（如 `--output /tmp/x.json` 会抛 ValueError）；需要绝对路径时请先 `cd` 到目标目录再运行。

### 2. analyze_content 元数据解析修复（最新提交）

原代码缺 `import re` 且 except 引用未定义变量，导致每次笔记元数据分析静默失败、keywords 恒为空。已修复并验证——这改变了"忠实复现对象"：原仓库在论文设置下实际从未产出有效元数据。论文对比时需声明此偏差来源。

完整细节见 `reports/stage0_audit.md` §5.0–5.1。

## STAGE 1 真实评测 runner（服务器使用）

`research/src/amem_forgetting/evaluation/stage1_runner.py` 是 STAGE 1 实验 1A 的正式载体：
LoCoMo 逐对话写入记忆（source_turn_ids=dia_id）→ KeepAll → raw 检索语义 QA →
Token-F1/EM/BLEU-1/ROUGE-L → run 目录契约 + gate。对抗题选项顺序按 question 哈希
确定性排列（修复未播种随机问题）。`run_id` 不可复用（raw append-only）。

```bash
# 服务器上：先装依赖，起模型服务（vLLM 示例，extension track 推荐）
pip install -r requirements.txt pytest jsonschema
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --port 8000

# smoke：1 个对话、每对话最多 5 题（--no-require_stage0_gate 仅供 dev 切片）
python research/src/amem_forgetting/evaluation/stage1_runner.py \
    --run_id stage1_smoke_$(date +%Y%m%dT%H%M) \
    --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
    --dataset data/locomo10.json \
    --limit_conversations 1 --max_qa_per_conversation 5

# pilot / full：放开 limit；gate PASS 后才进入 STAGE 2
```

后端支持 `vllm`（默认 `http://localhost:8000/v1`）、`ollama`（11434）、
`openai`（需 OPENAI_API_KEY）、`sglang`；统一走 `evaluation/backends.py` 适配层。
runner 会在开始前检查 stage0 gate 状态：BLOCKED 时拒绝运行（dev 切片可显式绕过并留痕）。

## 常用命令

```bash
# 测试（无需任何 ML 依赖）
python3 -m pytest research/tests/ -q

# wiring smoke（写 run 目录；run_id 需唯一——原始结果 append-only）
python3 research/scripts/stage1_mock_smoke.py --run_id stage1_mock_smoke_<your-suffix>

# 验证某次 run 的事件链完整性
python3 -c "import sys; sys.path.insert(0,'research/src'); \
from amem_forgetting.instrumentation.events import EventLog; \
log = EventLog.load('research/runs/raw/<run_id>/events.jsonl'); print(log.verify())"
```

## 关键约定

- `add_note` 必须传 `source_turn_ids`，否则拒绝写入（provenance 强制）。
- 事件按操作粒度成组（op_id）；op 内 write→link→evolve 有序，replay 以 op 为事务原子应用。
- 上游演化是覆盖式的；instrumentation 在观测层保留 before/after 版本与触发边，不改上游行为。
- 检索语义审计结论见 `reports/stage0_audit.md` §3：links 参与 QA（raw 路径 1-hop 扩展），
  事件日志以 `mode=flat_topk | raw_1hop_expansion` 显式分离两轨。
- 所有 run 文件名走白名单 + 原子写入；run_id 匹配 `[A-Za-z0-9][A-Za-z0-9._-]*`。

## 当前状态

- STAGE 0：**BLOCKED**（缺 LLM 后端与嵌入栈；fallback 方案见 `data_manifest/models.json`）
- STAGE 1 代码：就绪并通过 mock wiring 验证；正式运行（reference/extension track）待环境
