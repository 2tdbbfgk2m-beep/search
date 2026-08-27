# STAGE 0 审计报告：环境、仓库与数据

- 日期：2026-08-27
- 执行者：本地实验代理（ZCode）
- 计划书版本：`research/AMEM_graph_forgetting_experiment_plan.md` v1.0（上游存于 GitHub `2tdbbfgk2m-beep/search` 仓库）
- 结论：**gate = BLOCKED**（阻塞项见第 7 节；代码侧 instrumentation 已就绪并通过 wiring 验证）

## 1. 仓库定位与身份

| 项 | 值 |
|---|---|
| 本地路径 | `work/A-MEM` |
| 上游 origin | `https://github.com/WujiangXu/AgenticMemory.git`（A-MEM 论文官方 benchmark 仓库） |
| 复现基线 | `0c8039f`（本地 `main`）**+ 安全加固提交（见 §5.0；纯防御性，不改变任何实验行为与数值路径）** |
| 许可证 | MIT（允许修改与再分发；论文引用义务见原文 Citation 节） |
| 本次工作分支 | `research/stage1-instrumentation`（自 `main` 拉出） |
| 先行工作分支 | `feature/value-aware-forgetting`（8 个提交：价值状态、审计链、割点保护等；**不作为本次复现基线**，因为其 value-rerank 已改变检索行为） |

按计划书 §3.1"不强行覆盖上游仓库"的要求：本次实验逻辑改动全部位于 `research/` 扩展目录；对上游文件的唯一修改是用户批准的安全加固（Mimosa Git 门禁要求），不触碰任何检索/演化/写入行为。

## 2. 五组件审计（STAGE 0 必做清单）

| 组件 | 位置 | 状态 |
|---|---|---|
| writer（笔记构建） | `memory_layer.py::MemoryNote.analyze_content` + LLM | 已定位；**发现 bug，见 §5.1** |
| linker（链接生成） | `memory_layer.py::process_memory` 的 `strengthen` 动作 | 已定位；链接以**邻居列表位置索引**存储（见 §5.2 风险） |
| evolver（记忆演化） | `memory_layer.py::process_memory` 的 `update_neighbor` 动作 | 已定位；**直接覆盖**旧 context/tags，无版本保留 → 已由 instrumentation 补齐（EvolutionTracker） |
| answerer（问答） | `test_advanced.py` / `test_advanced_robust.py` | 已定位；依赖 LLM 后端 |
| embedding | `SimpleEmbeddingRetriever`（`all-MiniLM-L6-v2`） | 已定位；本地无 sentence-transformers，测试用确定性 stub 替代 |

## 3. 关键审计发现：links 是否参与 query-time retrieval

**结论：参与，但仅限 QA 上下文构建路径，且为 1-hop 扩展。**

证据链（`memory_layer.py`，commit `0c8039f`）：

```text
question
  -> test_advanced*.py: memory_system.find_related_memories_raw(content, k)
     -> SimpleEmbeddingRetriever.search(query, k)        # flat 嵌入余弦 top-k（种子）
     -> 对每个种子: 遍历 note.links（位置索引 -> notes 列表）
        -> 邻居内容直接 append 进 prompt 字符串            # 1-hop 图扩展，无深度限制到 k+1
  -> prompt -> LLM 回答
```

而 `find_related_memories`（无 `_raw` 后缀）**不做**链接扩展——它只用于演化前的邻居查找。

按计划书 §2.4 的决策规则：

- `links_used_in_retrieval = true`（QA 路径）→ 主结果可保留 graph 扩展分支，`REWIRE` 在 QA 轨有功能意义；
- 但扩展是**拼接进 prompt 的字符串**，无独立候选池、无多跳、无候选上限控制（每种子最多 k+1 个邻居，`j>=k` 的检查在 append 之后，属上游 quirk，已如实镜像进 instrumentation）；
- 忠实 flat 检索轨 = `SimpleEmbeddingRetriever.search` 的 top-k 种子集合；graph 扩展轨 = 种子 + 1-hop 邻居。两轨已在事件日志 `retrieve` 事件的 `mode`/`seed_ids`/`expanded_ids` 字段中显式分离。

## 4. 环境与模型审计

| 项 | 状态 | 影响 |
|---|---|---|
| Python | 3.13.9 | 上游 requirements 未声明版本范围；3.13 下 `ast.Str` 弃用告警（无害） |
| sentence-transformers / torch | **未安装** | 真实嵌入检索不可用；测试用确定性 hash 编码器 stub |
| litellm / openai | **未安装** | 真实 LLM 调用不可用 |
| OPENAI_API_KEY | **未设置** | reference track 无法启动 |
| Ollama / vLLM | **未安装/未运行** | extension track 无法启动 |
| numpy / scikit-learn / nltk | 已安装（2.3.5 / 1.7.2 / 3.9.2） | STAGE 3 logistic regression 可用 |
| pytest | 已安装（8.4.2） | 测试可运行 |

**数据**：`data/locomo10.json`（LoCoMo 10 个对话，2.8 MB）
SHA-256：`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
结构：list[10]，每项含 `qa` / `conversation` / `event_summary` / `observation` / `session_summary` / `sample_id`。LongMemEvalS **未获取**。

## 5. 上游代码缺陷清单（如实记录）

### 5.0 安全加固变更记录（本次唯一对上游的修改，经研究者批准）

Mimosa 静态扫描在 commit 门禁拦截出 6 处高危"路径穿越"：评测/缓存写入点直接用 argparse 传入的路径调用 `open(..., 'wb'/'w')`。加固内容（不改变任何实验行为）：

1. `memory_layer.py` 新增 `safe_output_path(path)`：拒绝 NUL 字节、绝对路径、解析后逃逸工作目录的相对路径。
2. 六个写入 sink 改为「BytesIO/StringIO 序列化 + `Path(safe_output_path(...)).write_bytes/write_text`」：
   - `memory_layer.py::HybridRetriever.save`、`SimpleEmbeddingRetriever.save`（pickle 缓存 + np.save 路径校验）
   - `test_advanced.py` 记忆缓存 pickle 与结果 JSON
   - `test_advanced_robust.py` 记忆缓存 pickle 与结果 JSON
3. 输出内容字节级不变；README 示例的全部相对路径用法保持可用；绝对路径输出从此被拒绝（如需支持须另行声明）。

未处理项：2 个低危"弱随机数"（对抗题协议用 `random.random()` 决定选项顺序，非加密用途）；评测缓存读取侧 `pickle.load` 属同源信任假设（本地单机研究工具），作为跨文件提示保留并在此记录。

### 5.1 analyze_content 的 JSON 解析崩溃（高影响，未修改）
### 5.1 analyze_content 的 JSON 解析崩溃【已修复，2026-08-27 研究者批准】

原缺陷：`memory_layer.py` 使用 `re.sub` 但模块未 `import re`；且 except 块引用未定义变量 `e`。实测（fake LLM 注入）后果：**每次笔记元数据分析都抛 NameError 并静默吞掉**，keywords 退化为 `[]`、context 退化为 `"General"`——在"论文设置"下笔记元数据全部是空壳。

修复内容：顶部补 `import re`；内层 except 改为绑定异常变量后如实打印。修复后经 fake LLM 验证：元数据正常解析（如 `keywords: ['Berlin', 'Alice', 'moved']`）。

**影响声明**：这是对"忠实复现对象"的一次变更——reference/extension 轨的复现基线从此为「0c8039f + 安全加固(§5.0) + 本修复」。论文数值对比时应说明此偏差来源：原仓库在该路径上实际从未产出有效元数据。真实 LLM 后端的首次 smoke 应包含对元数据质量的 sanity check（如 keywords 平均个数）。

其余未修复缺陷：

1. **links 用位置索引而非稳定 ID**：`suggested_connections` 返回邻居在 `list(self.memories.values())` 中的**位置**，存进 `note.links`。任何记忆删除/插入都会让旧链接指向错误对象。instrumentation 已通过 `MemoryRegistry.link_ids` 做稳定 ID 解析并记录 `unresolved_link_count`。
2. **演化覆盖无版本**：`update_neighbor` 直接覆盖旧 context/tags → 已由 `EvolutionTracker`（before/after + DAG）补齐。
3. **QA prompt 可能包含重复记忆行**：raw 路径遍历时同一记忆可既作种子又作他人邻居被重复 append（无去重），浪费上下文 token；instrumentation 的 `expanded_ids` 按多重集合忠实保留该现象。
4. **对抗题协议用未播种的 `random.random()` 决定选项顺序**：不同运行/不同 QA 重跑之间同一问题的选项排列会漂移，adversarial 指标带不可控方差（详见本次会话分析：k-sweep 只重跑 QA 时，缓存同一记忆状态下的 C5 分数仍会逐次漂移）。Mimosa 的"加密用途弱随机数"标签属规则误报（非加密用途），但实验确定性要求它必须在正式评估前改为按 question_id 派生的确定性排列或预注册 seed。
5. **`HybridRetriever.add_document` 引用未导入的 `torch`**（死代码路径，未被 A-MEM 主流程调用，未处理）。

## 6. STAGE 1 instrumentation 现状（代码已就绪）

全部位于 `research/src/amem_forgetting/`：

- `instrumentation/canonical.py`：确定性序列化 + 状态哈希
- `instrumentation/events.py`：append-only 哈希链事件日志（§3.9 schema + 篡改检测 + 追加写入保护）
- `instrumentation/registry.py`：memory 记录（source_turn_ids、稳定 ID、token/byte 预算字段）
- `instrumentation/evolution.py`：before/after 版本 + 演化 DAG（含环检测）
- `instrumentation/wrapper.py`：对未修改的 `AgenticMemorySystem` 做黑盒观测（write/link/evolve/retrieve/query/answer 全事件化；`retrieve` 事件显式分 `flat_topk` 与 `raw_1hop_expansion` 两轨）
- `instrumentation/replay.py`：任意 step 的状态重建 + 边界一致性验证
- `testsupport/`：依赖 stub（无 ML 栈环境导入上游模块）+ 确定性 FakeLLMController
- `schemas.py` / `runsupport.py`：JSON Schema 校验 + 计划书 §3.1 的 run 目录契约（原子写入、文件名白名单）

测试：`research/tests/test_stage1_instrumentation.py`，**20 项全过**，覆盖计划书 §13.4 中本阶段适用的项目（memory_id 稳定性、事件哈希链、snapshot replay、provenance、chronological emission、append-only）。

Mock wiring smoke：`runs/raw/stage1_mock_smoke_v1/`（fake LLM + stub embedding 驱动真实上游代码 5 轮 write→link→evolve + 3 问 retrieve→answer；gate PASS：replay 边界一致性 100%、在线/重放状态哈希一致、schema 0 错误）。**此为 wiring 验证，不产生任何论文数值。**

## 7. 阻塞项与 fallback（gate.json 依据）

**BLOCKED 项（进入正式 STAGE 1 前必须解决）：**

1. 无可用 LLM 后端（无 OPENAI_API_KEY、无 Ollama、无 vLLM）→ 端到端真实 smoke 无法运行。
2. 无 sentence-transformers → 真实嵌入检索无法运行。

**预声明 fallback（依计划书 §0 规则 4）：**

- 方案 A（推荐，extension track）：安装 Ollama + `qwen2.5:3b`（≈2GB）+ pip 安装 sentence-transformers → 全链路本地运行；
- 方案 B（reference track）：提供 OPENAI_API_KEY → `gpt-4o-mini`；
- 方案 C：远端服务器（vLLM）可用时按 README 的 vLLM 路径接入。

任一环境就绪后，重跑 `research/scripts/stage1_mock_smoke.py` 的真实后端版本即可完成 STAGE 0 的真实 smoke 项并解除 BLOCKED。

## 8. 给研究者的决策点

1. ~~**上游 §5.1 缺陷**：修还是不修？~~ **已决定并修复（2026-08-27）**：补 `import re` + except 绑定异常；复现基线相应更新为「0c8039f + §5.0 加固 + 本修复」。
2. **links 位置索引缺陷**：instrumentation 已在观测层解决（稳定 ID 解析），不触碰上游写入行为。若后续阶段需要真正稳定的链接，需要研究者批准修改上游（超出最小侵入范围）。
3. **对抗题未播种随机化（新增决策点，源自 §5 其余缺陷第 4 条）**：正式 confirmatory 前必须改为确定性排列（按 question_id 哈希派生）或预注册全局 seed；否则 H 系列假设检验在 C5 子集上不可复现。
