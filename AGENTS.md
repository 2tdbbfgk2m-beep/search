# AGENTS.md — 给服务器执行 AI 的操作说明

本仓库分支 `research/stage1-clean` 是 **A-MEM 图感知学习式遗忘研究**的工作分支
（干净根历史的孤儿提交，内容与开发分支 `research/stage1-instrumentation` 完全一致；
原历史因上游浅克隆损坏无法推送，故嫁接）。上游为 WujiangXu/AgenticMemory @0c8039f，
另有经批准的两处上游修改（路径安全加固 + analyze_content 修复），见下文"基线声明"。

## 第一优先级：阅读顺序

1. `research/AMEM_graph_forgetting_experiment_plan.md` — 总计划书（分阶段实验合同）
2. `research/README.md` — 目录结构、命令、上游修改记录
3. `research/reports/stage0_audit.md` — 环境审计报告 + gate 状态
4. 本文件 — 服务器侧执行入口

## 当前状态（2026-08-28）

- **STAGE 0**：代码侧完成，gate=BLOCKED（等真实 LLM 后端 smoke；你所在的机器就是解铃人）
- **STAGE 1 instrumentation**：完成，26 个测试全绿（`python3 -m pytest research/tests/ -q`）
- **STAGE 1 真实 runner**：完成待跑（`evaluation/stage1_runner.py`）
- 未做：任何遗忘策略（STAGE 2+）、scorer、图特征——按计划书 gate 规则推进

## 服务器上的第一步：解除 STAGE 0 BLOCKED

```bash
# 1. 环境自检（Python>=3.10 即可；GPU 信息记入报告）
pip install -r requirements.txt pytest jsonschema

# 2. 起模型服务（extension track 推荐；无 GPU 则用 ollama qwen2.5:3b）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct --port 8000 --max-model-len 8192

# 3. 跑测试确认代码完好（无需 LLM）
python3 -m pytest research/tests/ -q   # 应 26 passed

# 4. STAGE 0 真实 smoke：用 runner 的 dev 切片（绕过 stage0 gate 并留痕）
python research/src/amem_forgetting/evaluation/stage1_runner.py \
    --run_id stage0_backend_smoke_$(date +%Y%m%dT%H%M) \
    --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
    --dataset data/locomo10.json \
    --limit_conversations 1 --max_qa_per_conversation 5 \
    --no-require_stage0_gate

# 5. smoke 指标 sanity check 后，更新 research/runs/raw/stage0_audit/gate.json
#    （backend 环境 BLOCKED 项 → PASS），然后正式跑 STAGE 1：
python research/src/amem_forgetting/evaluation/stage1_runner.py \
    --run_id stage1_ext_qwen3b_$(date +%Y%m%dT%H%M) \
    --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
    --dataset data/locomo10.json \
    --dataset_sha256 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4
```

## 基线声明（写进任何报告/论文时必须保留）

复现基线 = **上游 0c8039f + 安全加固（路径校验）+ analyze_content 修复**。
其中 analyze_content 修复影响重大：原仓库在该路径上从未产出有效笔记元数据
（keywords 恒为空），论文数值对比时必须声明此偏差。详见
`research/reports/stage0_audit.md` §5.0–5.1 与 `research/README.md` 的修改记录。

## 硬性规则（违反即前功尽弃）

1. **run_id 不可复用**——RunDirectory 会拒绝写入非空目录（raw append-only）。
2. **gate PASS 前不得进入下一 stage**；runner 会检查 stage0 gate 并拒绝运行。
3. **不得静默换模型/数据/预算/seed**；一切差异记录进 run_manifest。
4. **对抗题选项顺序已确定性化**（question 哈希派生）——不要改回 random.random()。
5. 每次运行产出 8 个必需文件 + `events_<sample>.jsonl` 分段；缺文件 = gate FAIL。
6. `research/runs/raw/` 下的内容只追加不修改、不删除。

## 已知坑

- LoCoMo 数据集 `data/locomo10.json`（2.8MB，10 对话/1986 题）随仓库走；
  confirmatory 前需按计划书 §3.3 冻结完整版并重算 SHA-256。
- vLLM 的 JSON 约束走 `guided_json` 顶层参数；OpenAI 走原生 `response_format`
  （`backends.py` 已自动分派）。
- 上游 `memory_layer.py` 的 links 用位置索引——instrumentation 已在观测层解析为
  稳定 ID；不要在上游对象上直接做删除/插入然后还期望旧链接有效。
