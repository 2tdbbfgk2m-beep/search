# A-MEM 图感知学习式遗忘

## 分阶段实验计划书（服务器 AI 执行版）

版本：1.0  
日期：2026-08-27  
研究对象：以 A-MEM 为基础的、面向结构化且会演化的记忆系统的学习式遗忘  
执行方式：服务器 AI 按阶段执行；未通过当前阶段门槛时不得自动进入下一阶段

### 快速导航

先把第 0 节总指令交给服务器 AI；随后严格按 STAGE 0 -> STAGE 8 执行。第 13 节是统一运行模板，第 14-16 节是止损和人工检查规则。

- [总指令与执行顺序](#0-先给服务器-ai-的总指令)
- [全局实验合同](#3-全局实验合同)
- [STAGE 0-2：复现、动机和预算基线](#4-stage-0环境仓库和数据审计)
- [STAGE 3-5：学习式与图感知遗忘](#7-stage-3非-rl-的-learned-utilityflat-baseline)
- [STAGE 6-8：延迟归因、扩展动作和确认性实验](#10-stage-6延迟后悔与-memory-specific-credit)
- [统一执行模板、止损和最小版本](#13-服务器-ai-的统一执行模板)


## 0. 先给服务器 AI 的总指令

以下文字可以直接复制给服务器上的 AI。它是总控协议，不替代后面的阶段计划。

```text
你是本研究的实验执行代理。研究目标是：在 A-MEM 的链接与记忆演化机制上，研究有限记忆预算下的学习式遗忘；重点验证图结构信息和延迟后悔归因是否能减少错误遗忘。

严格遵守以下规则：
1. 一次只执行一个阶段。当前阶段未生成 gate.json 且 status=PASS，不得执行下一阶段。
2. 开始前先阅读仓库中的 README、AGENTS.md、现有测试和 git status；保留用户已有修改，不得 reset、checkout -- 或覆盖未知文件。
3. 不得把论文中的报告数值当作本地复现结果。所有数值必须来自本地原始日志和可重跑脚本。
4. 不得静默更换模型、数据划分、预算、检索 k、标签规则或随机种子。若环境不支持原设置，先记录差异，再使用预先声明的 fallback，并把结果与 reference track 分开。
5. 所有实验运行都必须记录：run_id、git commit、完整配置、模型与版本、数据版本/哈希、seed、开始结束时间、硬件、错误信息。
6. 原始结果只追加，不就地修改；汇总表和图必须由脚本从原始 JSONL/SQLite 生成。
7. 训练或推理阶段不得使用测试问题、未来答案、未来图结构、未来演化结果或测试集 gold evidence 作为决策特征。未来信息只能用于训练标签或事后评价，并必须在代码中明确标记。
8. 若结果不支持假设，报告 FAIL 或 INCONCLUSIVE，并给出错误分析；不得为了通过门槛而调阈值、删样本或挑选 seed。
9. 每个阶段结束时生成四类文件：results.jsonl、metrics.csv、error_analysis.md、gate.json；同时保存执行命令和配置快照。
10. 任何含真实凭据、个人信息或敏感交互的日志只保存在授权的本地路径；对外上传前必须脱敏。需要验证精确字符串时保存哈希、长度和局部掩码，不把秘密放进报告。

当前任务：只执行我明确指定的 STAGE。执行完后停止，输出：完成项、未完成项、命令、产物路径、资源消耗、gate 结论和下一阶段所需条件。
```

### 推荐执行顺序

```text
STAGE 0 环境与代码审计
  -> STAGE 1 忠实复现 A-MEM + 事件/血缘日志
  -> STAGE 2 证明增长、污染与预算下的遗忘需求
  -> STAGE 3 非 RL 的 learned utility（平面节点模型）
  -> STAGE 4 构造并冻结 A-MEM Forgetting Benchmark
  -> STAGE 5 图感知 learned forgetting
  -> STAGE 6 延迟后悔与 memory-specific credit / RL
  -> STAGE 7 可选的 ARCHIVE / MERGE / REWIRE
  -> STAGE 8 确认性全量实验、统计和论文产物
```

### 三种运行档位

服务器 AI 应为每一阶段提供以下三个档位，先跑 smoke，再跑 pilot，最后才跑 confirmatory。

| 档位 | 用途 | 数据规模 | seed | 预算点 |
|---|---|---:|---:|---|
| smoke | 检查代码、schema、端到端链路 | 1 个对话或每类 10 个合成样本 | 1 | 20%、60%、100% |
| pilot | 调试特征、标签和错误分析 | 3 个对话；每类 50-100 个合成样本 | 3 | 10%、20%、40%、60%、80%、100% |
| confirmatory | 支撑论文主张 | 全部预注册数据 | 至少 5 | 同 pilot |

若 API 成本或 GPU 时间过高，可以减少 smoke/pilot，但不得减少 confirmatory 的主要数据而不在报告中说明。

---

## 1. 研究问题、假设和主张边界

### 1.1 总研究问题

在 A-MEM 这种会建立链接、并且新记忆会改变旧记忆表示的系统中，如何在有限记忆预算下学习遗忘，使系统减少存储和检索成本，同时不删除未来任务所需的记忆，也不破坏关键的演化血缘与图结构？

### 1.2 子问题

1. **Q1：是否确实需要遗忘？** 记忆持续增长、陈旧/冲突/冗余信息增加时，keep-all A-MEM 是否出现成本上升、检索噪声或任务性能下降？
2. **Q2：未来有用性是否能被提前学习？** 在查询尚未出现时，基于写入时可见的特征预测未来 utility，是否优于 recency、FIFO、LRU、MemoryBank 等规则？
3. **Q3：图结构是否提供额外信息？** 一个节点自身很少被直接使用，但可能是连接簇、演化后代或多跳证据路径的关键；加入图和演化血缘特征后，是否能减少错误遗忘？
4. **Q4：长期后果如何归因？** 删除在很多步之后才造成失败时，memory-specific 或 graph-propagated credit 是否优于只给所有历史动作相同 terminal reward？

### 1.3 预注册假设

| 假设 | 最小支持证据 | 不能宣称的内容 |
|---|---|---|
| H1 遗忘需求 | 至少一个真实/合成设置中 keep-all 被成本或污染伤害，且预算策略可在非劣任务质量下减少资源 | 不能宣称记忆越多总是越差 |
| H2 learned utility | 平面 learned scorer 在至少两个预算点超过最强非学习规则，或在等准确率下显著节省预算 | 不能宣称自监督标签普遍等价于人工标签 |
| H3 graph awareness | 在相同节点特征、预算和答案模型下，图/血缘特征显著降低错误遗忘，主要增益出现在 bridge/evolution 子集 | 不能把图扩展轨结果写成原始 A-MEM 结果 |
| H4 delayed credit | memory-specific 或 graph-propagated credit 相比 terminal-only 在 held-out 长延迟任务上降低 regret/WFR，并保持任务质量 | 不能宣称已经解决完整因果归因 |

### 1.4 非目标

本项目第一阶段研究的是**功能性、预算驱动的遗忘**，不是隐私法规意义上的 machine unlearning。删除一个节点后，其他节点中已经写入的内容可能仍然含有它的影响；若要声称“完全消除演化影响”，必须另做可验证的 lineage rollback 和残留信息测试，不能用普通 DELETE 代替。

---

## 2. 论文依据与需要特别校准的地方

### 2.1 A-MEM 提供的基线

A-MEM 的记忆笔记包含原始内容、时间戳、关键词、标签、上下文描述、嵌入和链接集合；新记忆加入时先以嵌入相似度找 top-k 邻居，再由 LLM 生成链接，并可能演化邻居的上下文/关键词/标签。论文的检索公式仍然是对记忆嵌入做余弦相似度排序。

复现时固定记录：

- 笔记构建、链接生成、记忆演化三个 LLM 调用是否启用；
- `top_k_link` 和 `top_k_retrieval`；
- 嵌入模型（论文使用 `all-MiniLM-L6-v2`）；
- 基础模型、温度、结构化输出约束；
- 每个节点的版本和来源 turn。

### 2.2 CURATOR 可借鉴的部分

CURATOR 用净价值密度

\[
\rho(m)=\frac{\hat V(m)-\lambda\hat H(m)}{b(m)},
\qquad
\hat V(m)=p(m)q(m)a(m)
\]

统一处理 keep、share、trust，并强调单位字节成本、负迁移、抽象收益和 provenance harm。它是本项目的强规则型基线/特征来源，不应被简化成普通的 recency score。

### 2.3 LRE 可借鉴的部分

LRE 将单位按未来复用情况打标签，用小型、查询盲的 logistic regression 预测 relevance，再按 `score / cost` 在预算下贪心保留，并逐字抽取而非摘要。对话自监督标签使用后续答案 token overlap 阈值 0.4；代理轨迹标签使用后续 identifier 复用至少 3 次。我们的 A-MEM 版本必须把“turn -> note -> evolved descendant”映射显式记录，否则标签会错位。

### 2.4 关键架构审计

在 STAGE 1 必须回答：A-MEM 代码中的 `links` 是否实际参与问答时的候选生成或多跳遍历？

- 若答案为“否”：主结果只能称为 **lineage/topology-aware forgetting over faithful A-MEM retrieval**；`REWIRE` 不得作为原始 A-MEM 能力。
- 若答案为“是”：记录遍历规则、邻居扩展深度、候选上限和额外 token 成本，并把它作为独立 `graph-retrieval` 分支。
- 两个分支的结果不能混在同一张主表中。

---

## 3. 全局实验合同

### 3.1 代码与目录约定

不要强行覆盖上游仓库；优先建立独立扩展目录或分支。推荐目录：

```text
research/
  configs/
  data_manifest/
  src/amem_forgetting/
    instrumentation/
    policies/
    labels/
    graph/
    rl/
  scripts/
  schemas/
  runs/
    raw/
    derived/
  reports/
  figures/
  tests/
```

每次运行目录必须包含：

```text
run_manifest.json
config.yaml
git_commit.txt
stdout.log
stderr.log
results.jsonl
metrics.csv
gate.json
```

### 3.2 模型轨道

为了兼顾可复现和服务器可执行性，使用两条轨道：

1. **Reference track**：若资源允许，按 A-MEM 论文复现 GPT-4o-mini 以及一个论文中出现的本地模型；用于检查方向是否大致一致。
2. **Extension track（主实验）**：选择服务器可稳定运行的一个固定开放模型，推荐 `Qwen2.5-3B-Instruct`；写入、演化、回答和所有策略共享同一模型与解码设置。

若推荐模型不可用，服务器 AI 只能在 STAGE 0 报告后选择一个 fallback，并在所有表格中显式写出模型名；不得把不同模型的结果直接平均。

### 3.3 数据集优先级

| 优先级 | 数据 | 用途 |
|---|---|---|
| P0 | LoCoMo | A-MEM 主复现、长期对话和多跳问答 |
| P0 | 自建 A-MEM Forgetting Benchmark | 精确控制 delay/update/conflict/bridge/evolution |
| P1 | LongMemEvalS | 检验查询盲未来 utility 在另一种长程对话设置上的泛化 |
| P2 | AppWorld 或其他工具代理集 | 只有前面通过后，验证 load-bearing state 的跨域迁移 |

数据版本、下载日期、许可证、预处理脚本和 SHA-256 必须进入 `data_manifest/`。不能以网上“最新”版本替换已经冻结的 confirmatory 版本。

### 3.4 预算定义

主要预算以**活动记忆序列化 token 数**计，次要报告以字节数计；因为节点大小不同，不能只报告保留节点百分比。

对每个会话前缀，先运行 keep-all 得到预算基准 \(C_{full}\)，设预算比例

\[
b\in\{0.1,0.2,0.4,0.6,0.8,1.0\},\qquad B=bC_{full}.
\]

所有选择器在同一 \(B\) 下运行。若某方法采用最近窗口，窗口必须对所有相关方法一致，并单独报告窗口不计入预算还是计入预算。主对话实验默认不设置隐藏的 recent carve-out；代理迁移实验才可预注册 `last_k=5`。

检索 `top_k` 主设置固定为 10；`20/40/50` 只用于独立敏感性分析。不能一边改变遗忘策略一边为某个方法单独调 `top_k`。

### 3.5 基线集合

最小基线：

1. `KeepAll`：无遗忘，上界但可能超预算。
2. `Random`：同预算随机保留，至少 5 个随机种子。
3. `FIFO`：删除最老节点。
4. `LRU/Recency`：按最近访问或最近写入。
5. `MemoryBank-decay`：按年龄指数衰减；若无访问流，其排序与 recency 等价，必须说明。
6. `TFIDF-content`：不看查询的内容显著性。
7. `CURATOR-lite`：实现净价值密度的可复现实验版；若无法完整复现 provenance head，明确标为 `lite`。
8. `LRE-flat`：本项目的平面 learned utility。
9. `GraphForget`：加入拓扑/演化特征的版本。
10. `Oracle-evidence`：只作不可部署的上界，不参加“最佳可部署方法”排名。

### 3.6 数据泄漏与划分

- 对 LoCoMo：按完整 conversation 分组做 Leave-One-Conversation-Out 或 grouped K-fold，不能按 turn 随机切分。
- 对 LongMemEvalS：按 question/haystack 分组 5-fold；对跨问题重复 session 做 exact-text dedup guard。
- 任何 feature transformer、TF-IDF、scaler、阈值和校准器只在训练折拟合。
- 推理时只使用写入时或当前前缀可见的信息；future answer、gold evidence、后续链接和后续演化只能出现在训练标签或事后评估代码中。
- 测试集问题不得参与 scorer 训练、阈值选择和 early stopping。

### 3.7 统计协议

主结果至少报告均值、标准差或标准误、95% 置信区间和样本数。推荐：

- 对问答题目做 paired bootstrap 10,000 次；
- 对多个 seed 和 conversation 做分层 bootstrap；
- 多个预算点/多个指标的显著性检验使用 Holm 校正；
- 除 p 值外报告绝对差、相对差和效应量；
- 把“没有显著差异”写成 non-significant，不写成“相同”。

### 3.8 统一指标

**任务质量**

- `Token-F1`、EM（适合短答案）；
- `BLEU-1/ROUGE-L/METEOR/SBERT`（与 A-MEM reference track 对齐）；
- `MultiHop-F1`、`Temporal-F1`、`Adversarial accuracy`；
- 自建基准的程序化 exact accuracy；
- 若使用 LLM judge，固定 judge 模型、prompt、temperature，并保存判定输入输出。

**记忆选择质量**

\[
IMR@B=\frac{|\text{gold-required}\cap\text{kept}|}{|\text{gold-required}|}
\]

\[
WFR=\frac{|\text{later-needed}\cap\text{deleted}|}{|\text{deleted}|}
\]

\[
BMR=\frac{|\text{stale/redundant}\cap\text{deleted}|}{|\text{stale/redundant}|}
\]

同时报告 ROC-AUC、PR-AUC、Brier/ECE、Recall@budget 和 budget@80%-recall，避免只看最终 QA。

**资源**

- 活动 token、序列化字节、峰值 RAM、检索延迟、端到端延迟、LLM 调用次数和估计费用；
- scorer CPU 时间、units/sec；
- 图特征计算时间和额外存储。

**结构与演化**

- `GoldPathPreservation`：删除后仍可达的 gold-required path 比例，只在图检索分支使用；
- `EvolutionAncestorRecall`：未来任务所需节点的祖先血缘中仍保留或可解释重建的比例；
- `LineageResidualRate`：删除后下游节点仍含被删除源信息的比例；这是功能遗忘与完全 unlearning 的区分指标；
- `GraphEditCost`：删除、合并、重连引起的边变化数和额外 token/LLM 成本。

### 3.9 记忆与事件 schema

每一条 memory 至少记录如下字段；向量可单独存储，但必须有 hash：

```json
{
  "memory_id": "stable-id",
  "source_turn_ids": ["conv1:turn5"],
  "created_step": 5,
  "raw_content": "...",
  "keywords": [],
  "tags": [],
  "context": "...",
  "embedding_model": "...",
  "embedding_hash": "sha256:...",
  "link_ids": [],
  "parent_memory_ids": [],
  "evolution_ancestors": [],
  "version": 0,
  "content_hash": "sha256:...",
  "token_size": 0,
  "byte_size": 0,
  "active_state": "active"
}
```

每次写入、链接、演化、检索、打分、删除、归档、合并、重连和回答都写事件：

```json
{
  "event_id": "...",
  "run_id": "...",
  "step": 0,
  "event_type": "evolve|retrieve|evict|query|...",
  "memory_ids": [],
  "policy": "...",
  "score": null,
  "budget": null,
  "state_hash_before": "...",
  "state_hash_after": "...",
  "prev_event_hash": "...",
  "event_hash": "...",
  "future_label": null,
  "credit_status": "pending|resolved"
}
```

哈希链仅用于可复现性、版本完整性和后悔归因，不要把它包装成“使用了区块链共识”或安全证明。

---

## 4. STAGE 0：环境、仓库和数据审计

### 4.1 目的

确认服务器确实能运行 A-MEM 的原始流程，明确模型/API/GPU/数据差异，避免后续把环境故障误判成算法结论。

### 4.2 给服务器 AI 的阶段指令

```text
只执行 STAGE 0。不要实现遗忘算法，不要训练 scorer，不要生成论文结论。

1. 定位或克隆 A-MEM benchmark repo 与 production repo；先检查 git status、README、AGENTS.md、许可证和现有测试。
2. 检查 Python、CUDA/ROCm、GPU 显存、CPU/RAM/磁盘、可用模型、embedding 模型、API 连通性和数据路径。
3. 固定并记录 commit、依赖 lock、模型版本、tokenizer、embedding 版本、解码参数。
4. 验证 LoCoMo 数据格式、对话数、问题类别、gold answer/evidence 字段；验证是否有 LongMemEvalS。
5. 用一个最小样本跑通：写入 -> 链接 -> 演化 -> 检索 -> 回答，并保存原始 prompt/response 的脱敏副本。
6. 审计代码：links 是否参与 query-time retrieval；memory evolution 是否覆盖原对象；是否已有稳定 ID、来源 ID、版本号和删除接口。
7. 只输出审计报告和 gate.json；发现阻塞时停下，不要自行大改上游代码。
```

### 4.3 必做检查清单

- [ ] 仓库、commit 和许可证已记录。
- [ ] `AGENTS.md` 和现有测试已阅读。
- [ ] `python -m ... --help` 或等价入口可调用。
- [ ] 最小样本端到端成功，并能从日志重建一个 memory。
- [ ] 明确 writer、linker、evolver、answerer、embedding 五个组件。
- [ ] 明确 `links_used_in_retrieval: true/false`。
- [ ] 明确是否支持低温/确定性解码；若不支持，记录随机性来源。
- [ ] 数据版本和 SHA-256 已冻结。

### 4.4 产物

```text
reports/stage0_audit.md
data_manifest/datasets.json
data_manifest/models.json
configs/stage0_resolved.yaml
runs/stage0_smoke/
  run_manifest.json
  results.jsonl
  metrics.csv
  gate.json
```

### 4.5 通过门槛

`gate.json` 只有在以下条件全部满足时才写 `PASS`：

1. 能在不修改算法的情况下完成一次端到端 smoke run；
2. 能定位每个生成 memory 的来源 turn；
3. 能区分原始 A-MEM 轨和扩展轨；
4. 任何缺失的模型/数据/API 都有明确 fallback 或 `BLOCKED` 记录。

---

## 5. STAGE 1：忠实复现 A-MEM，并建立事件与演化血缘

### 5.1 目的

先复现 A-MEM 的写入、链接、演化和问答流程，再加观测能力；这一阶段不做策略性删除。最终要得到一个可以反事实重放的事件溯源系统。

### 5.2 给服务器 AI 的阶段指令

```text
只执行 STAGE 1。先保留原始 A-MEM 行为，再做最小侵入式 instrumentation。

1. 按 STAGE 0 的 resolved config 跑 reference track（可用时）和 extension track；所有方法先 KeepAll。
2. 为每个 memory 增加稳定 memory_id、source_turn_ids、version、content_hash、embedding_hash、token_size、byte_size。
3. 为 write/link/evolve/retrieve/query/answer 记录 append-only event log，并计算 prev_event_hash/event_hash。
4. 保存演化前后版本，不要只覆盖旧对象；记录 new_memory -> neighbor 的影响边和 prompt hash。
5. 实现 state snapshot/replay：给定事件日志可以重建任意 step 的 memory 集合、link 图和版本。
6. 审计 query-time retrieval 是否使用 link 图；若不使用，设置 faithful_flat_retrieval 分支，并另建 graph_retrieval 扩展开关但不要在本阶段启用。
7. 在 LoCoMo smoke 和完整 P0 数据上运行原始 A-MEM，生成与论文指标对齐的 baseline；不要求数值完全相同，但必须解释偏差。
8. 运行 replay 一致性测试和双次运行随机性测试。
```

### 5.3 实现要求

#### 记忆版本与血缘

当 A-MEM 的新记忆演化旧记忆时，旧对象不能物理丢失。至少保存：

```text
old_memory_id, old_version, new_version,
trigger_memory_id, neighbor_set_at_time,
changed_fields, before_hash, after_hash, event_id
```

`evolution_ancestors` 可先用有向无环图表示；如果实际存在循环链接，分别保存 `link_graph` 与 `evolution_dag`，不要强行合并为一张图。

#### 检索语义审计

必须在报告中给出以下伪代码或真实调用链：

```text
query -> embedding -> candidate generation -> top-k -> graph expansion? -> prompt
```

若 `graph expansion?` 为否，后续 `GoldPathPreservation` 只能作为结构诊断指标，不能作为原始 A-MEM 的 QA 因果解释。

### 5.4 实验与验收

| 实验 | 设置 | 主要输出 |
|---|---|---|
| 1A 端到端复现 | LoCoMo smoke/full，KeepAll | F1、BLEU-1、五类任务、token、延迟 |
| 1B LG/ME 消融 | full、w/o ME、w/o LG&ME | 与 A-MEM 表 3 对齐的比较 |
| 1C `k` 敏感性 | `k=10,20,40,50`，其余固定 | QA、检索 token、噪声 |
| 1D replay | 随机抽取至少 100 个事件边界 | snapshot hash、图和 memory 字段一致性 |
| 1E 双运行 | 同配置 2-5 次 | 非确定性分布、成本 |

### 5.5 通过门槛

- 复现轨可在同一数据版本下重复运行；
- replay 后状态 hash 与在线状态一致率 100%；
- 每个演化更新都能追溯到触发的新 memory；
- 论文 reference 与本地 extension 的差异有表格说明；
- 若无法满足，`gate=BLOCKED`，不得进入 learned scorer。

---

## 6. STAGE 2：证明“记忆增长/污染使遗忘有必要”

### 6.1 目的

这一步不追求新算法，只回答动机：keep-all A-MEM 是否在有限预算、长时延和污染环境中被成本或质量拖累。

### 6.2 给服务器 AI 的阶段指令

```text
只执行 STAGE 2。所有策略共享同一个写入器、演化器、回答器、数据顺序和预算；先运行 KeepAll 建立 full-memory 参照。

实现并运行以下四组实验：
1. Memory Growth：在固定目标事实和问题之间插入无关历史，测记忆规模、检索噪声和 QA。
2. Stale/Update：先写事实 v1，再写 v2，分别测试“当前事实”和“历史事实”问题，防止把所有旧事实都错误删除当成成功。
3. Redundancy/Conflict：加入可控数量的重复或冲突条目，记录污染比例和来源。
4. Budget curve：在 10/20/40/60/80/100% 的 token/byte 预算下比较 KeepAll、Random、FIFO、LRU、MemoryBank-decay、TFIDF-content、CURATOR-lite。

先跑 smoke，再跑 pilot；只有数据生成、gold evidence 映射和指标脚本通过测试后才跑 confirmatory。
```

### 6.3 实验设计

#### 2A Memory Growth

对每个目标问题保存一个固定 `target_evidence_set`，在 evidence 与 query 之间插入 `distractor_count ∈ {0,50,100,250,500}` 条无关或弱相关记忆。若真实对话长度不足，使用模板化合成 distractor，并在报告中分开标记 `real` 与 `synthetic`。

报告：

- active memory count/bytes/tokens；
- retrieval latency 与 answer prompt tokens；
- evidence recall、evidence precision、irrelevant retrieval ratio；
- QA F1/EM；
- 每新增 100 条记忆的边际成本和质量变化。

#### 2B Stale/Update

每个实体至少构造三种问题：

1. “现在/最新值是什么？”
2. “之前的值是什么？”
3. “值何时发生变化？”

gold 中保存有效时间区间和所需版本；不要只用“旧=坏、新=好”的单一标签。

#### 2C Redundancy/Conflict

- redundancy ratio：同义 paraphrase 数量 / 总条目数；
- conflict ratio：互相矛盾条目数量 / 总条目数；
- source reliability 只作为数据属性，不在第一版策略中偷偷使用。

#### 2D Budget curve

主预算按序列化 token；字节、节点数作为辅助。所有方法在同一 `B` 下运行；随机方法至少 5 seeds。

### 6.4 H1 通过门槛

满足以下任意一条，并有 95% paired bootstrap CI 支持，即可将 H1 标为 `PASS`：

1. 某一污染/长时延设置中，keep-all 的 QA 或 evidence precision 随污染显著下降；或
2. 某一预算点的可部署策略将活动记忆减少至少 40%，且 QA 相对 KeepAll 不低于 2 个百分点以内（预注册 non-inferiority margin）；或
3. keep-all 在资源指标上明显超出预算，而至少一个预算策略在质量不劣下满足预算。

如果所有设置都显示 keep-all 最优且无资源压力，写 `H1=NOT_SUPPORTED`，先检查任务是否真的需要长期记忆，不得直接进入 RL。

### 6.5 产物

```text
reports/stage2_motivation.md
reports/stage2_failure_cases.jsonl
figures/stage2_growth_*.png
figures/stage2_budget_pareto_*.png
runs/stage2_*/results.jsonl
runs/stage2_*/metrics.csv
runs/stage2_*/gate.json
```

---

## 7. STAGE 3：非 RL 的 Learned Utility（Flat Baseline）

### 7.1 目的

先不训练策略网络，验证“未来使用情况可以从写入时可见特征中学习”这一较小假设。这个阶段是图感知和 RL 的必要基线。

### 7.2 给服务器 AI 的阶段指令

```text
只执行 STAGE 3。实现一个查询盲的、节点级 learned utility scorer；禁止把未来问题或测试 gold evidence 作为在线特征。

1. 建立 turn -> A-MEM memory -> evolved descendant 的 provenance 映射。
2. 实现三种标签：gold-evidence（监督上界）、log-self-supervised（后续答案 overlap/retrieval/reuse）、counterfactual 小样本标签（可选）。
3. 只用 prefix 可见的节点特征训练 L2 logistic regression；TF-IDF 版本作为文本增强，不能把图特征混入本阶段主模型。
4. 按 conversation/question 分组交叉验证，所有 vectorizer/scaler/calibrator 只在训练折拟合。
5. 按 score/cost 在相同活动 token 预算下贪心保留，按时间顺序输出；不得摘要或改写记忆。
6. 与所有 STAGE 2 基线在相同 seed、预算、答案模型和检索 k 下比较。
7. 先跑标签质量和 scorer AUC，再跑下游 QA；如果标签对齐失败，先做错误分析，不得直接扩大模型。
```

### 7.3 标签定义

为每个 memory \(m_i\) 在写入时刻 \(t_i\) 定义未来窗口 \(H\)。主标签如下：

#### Gold-evidence（监督上界）

\[
y_i^{gold}=1
\quad\text{iff}\quad
source(m_i)\cap GoldEvidence(q_{future})\neq\varnothing
\]

如果未来问题需要一个已经演化的后代节点，则其祖先通过 `evolution_dag` 回溯；报告 direct evidence 和 ancestor evidence 两种版本。

#### Log-self-supervised（部署可获得）

参考 LRE，但适配 A-MEM：

- 后续正确回答中，memory 被检索且其来源/后代出现在证据对齐中；或
- memory 内容与后续答案 normalized token overlap \(\ge 0.4\)；或
- 在代理迁移轨中，它引入的 identifier 在未来至少复用 3 次。

这些条件分别记录，不要不加说明地 OR 成一个不可解释标签。主结果使用预注册组合；其他组合是消融。

#### Counterfactual utility（可选）

在训练折中，对少量问题比较保留/屏蔽某 memory 时的答案正确性或 judge 分数差：

\[
\Delta_i=Score(answer\mid M)-Score(answer\mid M\setminus\{m_i\}).
\]

它可以作为连续目标或错误分析，不应在测试时调用昂贵的反事实 LLM。

### 7.4 节点特征

主模型只使用在时刻 \(t_i\) 可见的特征：

```text
age / recency
write position
raw/context/keyword/tag token length
retrieval count up to t_i
successful retrieval count up to t_i
update/evolution count up to t_i
content novelty / nearest semantic similarity
duplicate score
contradiction flag available up to t_i
source type and validity interval if explicitly present
```

文本增强版本加入训练折拟合的 TF-IDF；不得加入 degree、betweenness、descendant utility 等图特征，这些留到 STAGE 5。

### 7.5 模型与选择器

主模型：L2 logistic regression，balanced class weight，输出 \(p_i=P(y_i=1\mid x_i)\)。

对每个候选节点按

\[
d_i=\frac{p_i}{cost_i}
\]

降序排列，在预算 \(B\) 下贪心保留。另实现一个精确/近似 0-1 knapsack 版本，确认贪心误差不会主导结论。

可选模型：小型 MLP 或 XGBoost，只作容量敏感性分析；若它超过 logistic regression，必须报告参数量、训练时间和额外部署成本。

### 7.6 实验矩阵

| 实验 | 比较 | 目的 |
|---|---|---|
| 3A 标签质量 | gold vs self-supervised；AUC/PR-AUC/正例率 | 判断日志标签是否有信号 |
| 3B flat scorer | node-only LR vs TF-IDF+node LR | 判断文本特征贡献 |
| 3C 预算曲线 | learned vs Random/FIFO/LRU/decay/CURATOR-lite | 判断 matched-budget 质量 |
| 3D 校准 | raw probability vs calibrated probability | 判断分数是否能跨预算使用 |
| 3E 跨对话泛化 | grouped holdout | 防止同一说话人/主题泄漏 |

### 7.7 H2 通过门槛

`H2=PASS` 需要同时满足：

1. self-supervised scorer 的宏平均 AUC 明显高于 0.5，且在至少一个主数据集上 PR-AUC 优于最强非学习规则；
2. 在至少两个预算点，learned scorer 的 QA 或 gold evidence recall 比最强非学习规则提高至少 2 个百分点，或在相同 QA 下预算节省至少 10%；
3. 结果在 held-out conversation 上成立，不能只在训练折成立。

若只有 gold 标签版本有效而 self-supervised 失败，保留两者并将 H2 标为 `PARTIAL`；后续可继续图感知，但不能宣称 annotation-free。

### 7.8 产物

```text
src/amem_forgetting/labels/
src/amem_forgetting/policies/flat_utility.py
reports/stage3_label_quality.md
reports/stage3_flat_utility.md
figures/stage3_auc_*.png
figures/stage3_pareto_*.png
runs/stage3_*/
```

---

## 8. STAGE 4：构造并冻结 A-MEM Forgetting Benchmark

### 8.1 目的

LoCoMo 能测试长期问答，但不能单独证明“图桥梁”和“演化影响”是删除后果。因此建立一个小而可审计的控制基准，在进入图模型前冻结测试集。

### 8.2 给服务器 AI 的阶段指令

```text
只执行 STAGE 4。构造程序化、可重放、带显式 gold provenance 的 benchmark；不要让 LLM 自由生成唯一真值。

1. 从固定事实模板和少量受控 paraphrase 生成 Delay、Update、Conflict、Redundancy、Bridge、Evolution 六类场景。
2. 每个样本保存事件序列、gold answer、gold evidence nodes、gold lineage/path、污染类型和难度。
3. 用程序化规则验证答案与证据；随机抽取至少 10% 做独立审计。
4. 生成 dev/pilot/test 三个不重叠 split；按模板、实体和 paraphrase 去重。
5. 对 Bridge 先验证当前检索是否真正使用 links。若不是，建立 graph-retrieval 扩展轨并把它标为额外架构因素。
6. 计算数据 manifest 和 checksum，冻结 test；冻结后不得因结果不好而改 test。
```

### 8.3 场景定义

| 场景 | 构造 | 主要测试 |
|---|---|---|
| Delay | 写入关键事实后插入 50/100/250/500 条 distractor，再提问 | 长时延保留 |
| Update | v1 -> v2，分别问 current/previous/change-time | 陈旧记忆处理，避免误删历史事实 |
| Conflict | 可信事实与冲突事实按来源/时间交错写入 | 冲突与负迁移 |
| Redundancy | 同一事实的多种 paraphrase 和近重复 | 压缩/去冗余 |
| Bridge | A -> B -> C，问题要求组合路径 | 节点价值与结构价值差异 |
| Evolution | 新节点触发旧节点 context/tag 演化，再删除源节点并问 downstream | 演化血缘与残留影响 |

Bridge 场景必须分成两种：

1. `flat-retrieval bridge`：只检验图特征是否能预测多跳所需节点，不能声称删除后图断开导致 QA 失败；
2. `graph-retrieval bridge`：先语义召回 seed，再按已记录 links 扩展；只有在此轨中报告 path reachability 和 REWIRE。

### 8.4 推荐规模

| 档位 | 每类样本 | paraphrase | 用途 |
|---|---:|---:|---|
| smoke | 10 | 1 | 端到端检查 |
| pilot | 50-100 | 2-3 | 找模板/标签错误 |
| confirmatory | 200-300 | 3-5 | 统计主结果 |

如果服务器资源有限，先减少 paraphrase，不要删除整个场景类别。

### 8.5 关键数据字段

```json
{
  "scenario_id": "evolution_0001",
  "scenario_type": "delay|update|conflict|redundancy|bridge|evolution",
  "event_sequence": [],
  "gold_answer": "...",
  "gold_evidence_memory_or_source_ids": [],
  "gold_lineage_edges": [],
  "gold_paths": [],
  "difficulty": "easy|medium|hard",
  "split": "dev|pilot|test",
  "template_hash": "sha256:..."
}
```

### 8.6 通过门槛

- 程序化答案/证据校验通过率 100%；
- 10% 独立审计样本中没有系统性 gold 错位；
- 事件重放能重建同一 memory/evolution graph；
- test manifest 已冻结；
- Bridge 的 retrieval 语义已明确标注。

若任何一项失败，`gate=BLOCKED`，先修 benchmark，不得训练 GraphForget。

---

## 9. STAGE 5：图感知 Learned Forgetting

### 9.1 目的

在 STAGE 3 的平面 scorer 上增加当前前缀可见的拓扑和演化血缘特征，严格验证图信息是否减少错误遗忘。

### 9.2 给服务器 AI 的阶段指令

```text
只执行 STAGE 5。实现 node-only、topology-aware、evolution-aware 三个容量可比的 scorer；主模型优先使用同一 logistic regression。

1. 从当前 prefix 的 link graph 和 evolution DAG 增量计算特征；不得读取未来边、未来 descendants 或测试 gold path。
2. 先比较 FlatUtility 与 GraphUtility，保持模型、预算、检索、答案模型完全相同。
3. 分别在 faithful_flat_retrieval 和 graph-retrieval（若存在）两条轨上测试，不能合并。
4. 做 w/o topology、w/o evolution、w/o neighbor utility、w/o cost normalization 消融。
5. 输出节点级错误案例：自身 utility 低但结构 utility 高、结构高但实际无用、演化残留和错误 centrality。
```

### 9.3 图和血缘特征

#### 拓扑特征（只基于当前图）

```text
degree / weighted degree
component size
approximate betweenness / articulation flag
PageRank 或局部 centrality
cluster boundary ratio
neighbor score mean/max（只能用当前已训练 scorer 或 prefix statistics）
semantic link confidence / link age
```

如果图规模较大，betweenness 使用固定采样点或局部近似，并报告计算复杂度和误差；不能每个预算点用不同近似。

#### Evolution features

```text
in/out evolution degree
descendant count within 1/2 hops
weighted descendant utility（训练/当前统计，不含未来 gold）
number of versions
time since last evolution
fraction of fields changed by evolution
ancestor/descendant content hash relation
```

推荐定义一个诊断量：

\[
ED_i=\sum_{j\in Desc(i)}w_{ij}\,u_j,
\]

其中 \(u_j\) 只能是当前可见或训练得到的 utility 估计；事后用 gold utility 计算的版本只能叫 `oracle diagnostic`。

### 9.4 模型组

| 模型 | 特征 | 作用 |
|---|---|---|
| F0 Flat | STAGE 3 节点特征 | 主对照 |
| F1 Topology | F0 + link topology | 测结构增益 |
| F2 Evolution | F1 + evolution DAG | 测 A-MEM 特有血缘增益 |
| F3 Oracle graph | 加事后 gold path/utility | 仅诊断上限，不可部署 |
| GNN（可选） | GraphSAGE/小型 message passing | 只有 F1/F2 有信号后再做 |

F0/F1/F2 主实验使用相同正则化、相同训练折和近似相同参数量；否则无法把增益归因于图信息。

### 9.5 主要实验

1. **Matched-budget comparison**：10%-100% 预算下 F0/F1/F2 的 QA、IMR、WFR、BMR、资源。
2. **Node utility vs graph utility**：专门抽取自身特征低但 degree/bridge/ED 高的节点，报告保留率和后续 QA。
3. **Evolution deletion**：比较 content-only DELETE 与 lineage-aware score；报告 downstream QA 和 `LineageResidualRate`。
4. **Graph retrieval factor**：在 graph-retrieval 轨比较是否扩展邻居、是否保留桥节点、是否重连。
5. **Scaling**：记忆规模、边数、图特征计算时间和 scorer 延迟。

### 9.6 H3 通过门槛

以下条件全部满足才写 `H3=PASS`：

1. 在相同预算和相同答案模型下，F2 相比 F0 在 evolution/bridge 子集的 WFR 至少相对下降 20%，或多跳/演化 QA 提高至少 5 个百分点；
2. 在普通 LoCoMo QA 上没有超过预注册的 1 个百分点回退，或回退有明确的预算/安全收益；
3. 增益在 held-out 模板/对话上成立，且不是只由更大的模型容量造成；
4. `w/o evolution` 在 evolution 子集上可观测下降，`w/o topology` 在 bridge 子集上可观测下降；
5. 若使用 graph-retrieval，报告其额外检索成本，并与 faithful-flat 结果分表。

如果只有合成 benchmark 有增益，标为 `H3=PARTIAL`，论文主张限制为“在受控演化场景中有效”。

### 9.7 产物

```text
src/amem_forgetting/graph/features.py
src/amem_forgetting/policies/graph_utility.py
reports/stage5_graph_aware.md
reports/stage5_casebook.jsonl
figures/stage5_ablation_*.png
figures/stage5_graph_vs_flat_*.png
runs/stage5_*/
```

---

## 10. STAGE 6：延迟后悔与 Memory-specific Credit

### 10.1 目的

研究一次删除在很久以后才造成失败时，如何把负 credit 归因到真正相关的删除，而不是把同一个 terminal reward 粗暴地广播给整段轨迹。

这一阶段必须建立在 STAGE 2、3、4 已通过，最好 STAGE 5 至少 `PARTIAL` 之后；否则 RL 很容易把数据或检索问题误学成策略。

### 10.2 给服务器 AI 的阶段指令

```text
只执行 STAGE 6。先做离线/缓存结果的策略学习，再考虑在线 LLM 环境；禁止每个梯度步重新调用昂贵 LLM。

1. 建立 eviction decision replay buffer：保存删除时 state snapshot、候选节点、动作 log-prob、预算、图 hash 和未来待解析标记。
2. 实现 KEEP/DELETE 的 set-level policy；有 variable token cost 时使用 masked top-k / knapsack selection，严格满足预算。
3. 用同一训练/验证/测试 split 比较四种 credit：terminal-only、discounted-terminal、memory-specific、graph-propagated。
4. 未来问题到达时才解析 regret：gold evidence、counterfactual delta、lineage/path 是否指向此前删除。
5. 将 regret 写回历史 deletion event；不得修改原始日志。
6. 先在程序化 benchmark 上用缓存的确定性 reward 训练，再在 held-out LoCoMo 上验证。
7. 对每一种 credit 做同样的 seed、预算和超参数搜索；超参数只在 dev 选择。
```

### 10.3 离线环境

将每个会话/场景表示为：

```text
prefix state s_t
candidate memories M_t
cost c_i
budget B_t
future query/evidence stream Q_{>t}
cached answer outcomes or programmatic evaluator
```

训练时不必每次调用答案模型：先为固定策略和关键反事实生成缓存，记录缓存模型版本；最终确认性实验仍需用冻结的真实 answerer 重跑。

### 10.4 策略与动作

第一版动作只允许：

\[
A=\{KEEP,DELETE\}.
\]

策略输出每个候选节点的 logit，再用 masked top-k / cost-aware selection 形成集合；所有超预算节点强制 mask。随机探索只在训练阶段使用，测试阶段采用确定性选择和单独报告采样结果。

推荐的最小实现顺序：

1. contextual ranking + supervised regret label；
2. offline REINFORCE/actor-critic；
3. 若确有长时延收益，再尝试 PPO/GRPO 风格更新。

不要一开始训练 LLM policy。研究对象是记忆控制器，先保持控制器小、可解释、可在 CPU 上运行。

### 10.5 奖励

基础终局奖励：

\[
R_T=Q_T-\lambda_c C_T,
\]

其中 \(Q_T\) 是 QA/程序化任务质量，\(C_T\) 是活动记忆或延迟成本。若预算是硬约束，主实验可令 \(C_T\) 只作报告，避免重复惩罚。

#### Credit variant A：Terminal-only

把 \(R_T\) 广播给本次所有选择动作。

#### Credit variant B：Discounted-terminal

\[
A_t=\gamma^{T-t}R_T.
\]

预注册 \(\gamma\) 搜索范围，只在 dev 调参。

#### Credit variant C：Memory-specific

若未来 gold evidence 或可靠 counterfactual 指向曾被删除的 \(m_i\)，则：

\[
r_i^{regret}=-\Delta_i\quad(\Delta_i>0),
\]

若删除 stale/redundant 节点没有造成损失并节省预算，可给非负局部奖励，但必须和任务奖励分开报告。

#### Credit variant D：Graph-propagated

若未来任务依赖一条 lineage/path，沿当前已记录的祖先或桥路径传播：

\[
r_i^{graph}=-\sum_{j\in GoldPath}\alpha^{dist(i,j)}\,\Delta_j/Z.

\]

只允许沿删除时已存在的边传播；未来才出现的边不能反向泄漏到当时 state。传播规则、\(\alpha\)、归一化和 path 截断必须记录。

### 10.6 后悔定义

```text
wrong_forget = deleted node is later required and no surviving equivalent exists
forgetting_regret = QA(full-oracle) - QA(actual-retained)
lineage_regret = downstream node becomes insufficient because a deleted ancestor was needed
```

每个 regret 事件都要带：最初 deletion_event_id、未来 query_id、证据/路径、置信度、解析方法和解决时间。

### 10.7 比较与消融

| 组别 | 目的 |
|---|---|
| F2 scorer（无 RL） | 检查 RL 是否真的增加价值 |
| terminal-only | 最弱 credit 基线 |
| discounted-terminal | 常见延迟奖励基线 |
| memory-specific | 测试直接证据归因 |
| graph-propagated | 测试演化/路径归因 |
| shuffled-credit | 负对照，检验是否依赖正确归因 |
| no-regret-buffer | 测试 delayed buffer 的必要性 |

### 10.8 H4 通过门槛

在 held-out 长延迟/演化测试上，至少满足：

1. memory-specific 相比 terminal-only 的 WFR 或 forgetting regret 相对下降 15% 以上；
2. graph-propagated 在 evolution/bridge 子集再带来可重复增益，且普通 QA 不显著回退；
3. `shuffled-credit` 不应产生同等增益；
4. 训练与评估使用不同场景/对话，且结果有 95% CI。

若只有离线缓存环境有效，标为 `H4=OFFLINE_ONLY`；不得声称已在真实在线代理中解决 credit assignment。

### 10.9 产物

```text
src/amem_forgetting/rl/replay_buffer.py
src/amem_forgetting/rl/credit.py
src/amem_forgetting/rl/policy.py
reports/stage6_delayed_credit.md
reports/stage6_regret_casebook.jsonl
figures/stage6_credit_comparison_*.png
runs/stage6_*/
```

---

## 11. STAGE 7：可选动作 ARCHIVE / MERGE / REWIRE

### 11.1 启动条件

只有在 `H1=PASS` 且 `H2=PASS`，并且 `H3` 至少 `PARTIAL` 时启动；否则这部分属于范围膨胀，应暂缓。

### 11.2 给服务器 AI 的阶段指令

```text
只执行 STAGE 7（若前置 gate 不满足则输出 NOT_STARTED）。先实现可逆 ARCHIVE，再实现 provenance-preserving MERGE；REWIRE 只有在 graph-retrieval 轨已验证后才实现。

1. ARCHIVE：从 active retrieval 移到 cold store，保留可恢复 ID；测试恢复成本和错误恢复率。
2. MERGE：只合并经过阈值和证据一致性检查的冗余节点；保留所有 source_ids 和 lineage。
3. REWIRE：把被删桥节点的路径做受控 contraction，不能凭 embedding 自动制造语义边；记录 contracted_via 元数据。
4. 在相同预算下比较 KEEP/DELETE 与多动作策略；报告每个动作的错误类型和额外成本。
```

### 11.3 重要边界

- `ARCHIVE` 是软遗忘，不是删除；必须报告 cold retrieval 的潜在泄漏。
- `MERGE` 的摘要/合并不能丢失精确 ID、时间和来源；先做 extractive/provenance-preserving 版本。
- `REWIRE` 只在 links 真正参与检索时有明确功能意义；在 faithful flat retrieval 轨不得把它当作主方法。
- 若研究目标转为“消除演化影响”，需增加 lineage rollback：从受影响下游节点重算 context/tag，并用残留信息 probe 测试，而不是仅删除源节点。

### 11.4 评价

除前述指标外报告：恢复成功率、merge fidelity、rewire path reachability、额外 LLM 调用、结构编辑数和 rollback 时间。

通过门槛是多动作策略在至少一个预算点达到与 F2/F4 相当的 QA，同时显著改善资源或可恢复性；否则把结果作为失败的扩展实验，不纳入主贡献。

---

## 12. STAGE 8：确认性全量实验与论文级交付

### 12.1 给服务器 AI 的阶段指令

```text
只执行 STAGE 8。先锁定代码、数据、配置和模型，不再进行探索性调参；所有主表由原始结果自动生成。

1. 按预注册矩阵跑 LoCoMo、冻结的 Forgetting Benchmark 和 LongMemEvalS；AppWorld 只有在 P0/P1 通过后作为可选跨域实验。
2. 跑全部基线、F0/F1/F2、credit ablation 和必要的 graph-retrieval 对照。
3. 用 paired/hierarchical bootstrap 生成 CI，使用 Holm 校正；保存统计脚本和随机数种子。
4. 生成主表、预算 Pareto 曲线、增长曲线、图/血缘消融、错误案例和资源表。
5. 生成 claim-evidence matrix：每个论文句子指向具体 run_id、指标和置信区间。
6. 冻结 raw runs 后再写报告；若主假设不成立，按实际结果改写结论，不删除负结果。
```

### 12.2 主确认性矩阵

| 维度 | 主设置 |
|---|---|
| 数据 | LoCoMo + 自建基准；LongMemEvalS P1 |
| 模型 | extension track 单一固定开放模型；reference track 单列 |
| 预算 | 10/20/40/60/80/100% token，bytes 辅助 |
| seed | 主方法至少 5；Random 至少 5；昂贵 reference 说明实际数 |
| 基线 | KeepAll、Random、FIFO、LRU、MemoryBank、TFIDF、CURATOR-lite、LRE-flat、GraphForget |
| 主要消融 | w/o topology、w/o evolution、w/o cost、w/o self-supervision、w/o regret |
| 统计 | 95% CI、paired bootstrap、Holm 校正、效应量 |

### 12.3 最终交付物

```text
reports/final_experimental_report.md
reports/claim_evidence_matrix.csv
reports/reproducibility.md
figures/main_*.png
tables/main_*.csv
tables/appendix_*.csv
data_manifest/final_manifest.json
configs/final_*.yaml
scripts/reproduce_stage*.sh
```

### 12.4 论文主张的最低证据映射

| 可能主张 | 必须有的证据 |
|---|---|
| A-MEM 需要预算型遗忘 | STAGE 2 growth/pollution + 资源/质量 Pareto |
| learned utility 有效 | STAGE 3 held-out AUC/QA + matched-budget baseline |
| 图感知减少错误遗忘 | STAGE 5 topology/evolution ablation + controlled subset |
| 延迟 credit 有效 | STAGE 6 credit comparison + shuffled negative control |
| 可部署 | scorer latency、额外内存、LLM calls、跨 seed 稳定性 |
| 完全消除演化影响 | 只有额外 lineage rollback/residual probe 通过才允许；否则不得写 |

---

## 13. 服务器 AI 的统一执行模板

每个阶段都按下列顺序执行，不要直接从 notebook 手工复制结果：

```text
1. git status + 读取上一阶段 gate
2. 解析 config，生成 config_hash 和 run_id
3. 运行 preflight（数据、模型、显存、磁盘、随机种子）
4. 运行 smoke
5. 运行单元测试和 schema 校验
6. 运行 pilot
7. 生成 pilot error_analysis；只有 gate 子项通过才运行 confirmatory
8. 追加原始 results.jsonl
9. 用固定脚本生成 metrics.csv 和 figures
10. 运行统计脚本
11. 写 gate.json，停止并等待下一阶段指令
```

### 13.1 `gate.json` 最小格式

```json
{
  "stage": "stage5",
  "status": "PASS|FAIL|PARTIAL|BLOCKED|NOT_STARTED",
  "run_ids": [],
  "preconditions": {},
  "criteria": {
    "criterion_1": {"value": null, "threshold": null, "pass": false}
  },
  "uncertainties": [],
  "blocking_issues": [],
  "next_stage_allowed": false,
  "generated_at": "2026-08-27T00:00:00Z"
}
```

### 13.2 配置模板

```yaml
project:
  name: amem_graph_forgetting
  track: extension  # reference | extension
  git_commit: auto
  run_root: runs

models:
  writer: Qwen2.5-3B-Instruct
  linker: Qwen2.5-3B-Instruct
  evolver: Qwen2.5-3B-Instruct
  answerer: Qwen2.5-3B-Instruct
  judge: null
  embedding: all-MiniLM-L6-v2
  temperature: 0.0

data:
  primary: LoCoMo
  secondary: null
  manifest: data_manifest/datasets.json
  split: grouped
  dedup_exact_text: true

amem:
  enable_link_generation: true
  enable_memory_evolution: true
  top_k_link: 10
  top_k_retrieval: 10
  retrieval_mode: faithful_flat  # faithful_flat | graph_expansion
  graph_hops: 0

forgetting:
  unit: amem_note
  cost: serialized_tokens
  budgets: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
  recent_protected_k: 0
  selection: density_greedy
  score: flat_utility
  actions: [KEEP, DELETE]

training:
  scorer: logistic_regression
  class_weight: balanced
  regularization: [0.01, 0.1, 1.0, 10.0]
  label_mode: self_supervised
  future_horizon: all
  fit_only_on_train_fold: true

evaluation:
  seeds: [1, 2, 3, 4, 5]
  bootstrap_reps: 10000
  confidence_level: 0.95
  multiple_testing: holm
```

### 13.3 结果记录原则

`results.jsonl` 每行代表一个最小可分析单元（一个 question、scenario、seed、policy、budget），而不是只写一个汇总数字。至少包含：

```json
{
  "run_id": "...",
  "seed": 1,
  "dataset_group": "conv_03",
  "question_id": "...",
  "policy": "GraphForget-F2",
  "budget_ratio": 0.4,
  "active_tokens": 0,
  "active_bytes": 0,
  "kept_memory_ids": [],
  "deleted_memory_ids": [],
  "gold_evidence_ids": [],
  "answer": "...",
  "gold_answer": "...",
  "task_metrics": {},
  "selection_metrics": {},
  "graph_metrics": {},
  "latency_ms": {},
  "error": null
}
```

### 13.4 必做自动测试

```text
test_memory_id_stability
test_event_hash_chain
test_snapshot_replay
test_no_future_feature_access
test_group_split_no_overlap
test_budget_never_exceeded
test_chronological_emission
test_verbatim_extraction_for_LRE
test_source_to_note_provenance
test_evolution_dag_acyclic_or_explicit_cycle_handling
test_graph_feature_uses_prefix_only
test_raw_results_append_only
```

---

## 14. 优先级和止损策略

### 必做（P0）

1. STAGE 0-3；
2. LoCoMo 主实验；
3. 自建 Delay/Update/Evolution 最小基准；
4. F0 vs F2 图/血缘消融；
5. 统计、错误案例和可复现脚本。

### 应做（P1）

1. 完整六类自建基准；
2. LongMemEvalS 外部验证；
3. STAGE 6 delayed credit。

### 可选（P2）

1. AppWorld 跨域代理实验；
2. GNN；
3. ARCHIVE/MERGE/REWIRE；
4. 真实硬件能耗/无线共享。

如果时间或算力不足，按 P0 -> P1 -> P2 截止；不要把 P2 的复杂动作塞入 P0 主实验。

### 明确停止条件

- STAGE 1 无法稳定重放 memory/evolution state；
- STAGE 2 没有任何可观测的预算/污染问题；
- STAGE 3 learned scorer 在 held-out 上接近随机；
- STAGE 4 gold provenance 无法可靠生成；
- STAGE 5 图特征增益只来自未来泄漏或更大模型；
- STAGE 6 credit 结果对 seed 极端敏感且无 CI 支持。

触发停止条件时，服务器 AI 应输出“当前结论、证据、最小修复建议”，不要自动扩大范围。

---

## 15. 最小可行研究（算力不足时）

如果服务器资源很有限，执行以下缩减版仍能回答核心问题：

```text
数据：LoCoMo 3 个对话 + 自建 Delay/Bridge/Evolution 各 50 个样本
模型：一个固定本地 3B 模型
预算：20%、40%、60%、100%
seed：3（confirmatory 再增到 5）
基线：KeepAll、FIFO、LRU、CURATOR-lite、LRE-flat
主比较：F0 Flat vs F2 Evolution-aware
credit：terminal-only vs memory-specific
不做：GNN、跨代理 SHARE/TRUST、真实硬件、复杂 REWIRE
```

最小版的论文故事只能写成：

> 在受控 A-MEM 长程对话和演化场景中，节点未来 utility 学习与演化血缘特征是否能在匹配预算下减少错误遗忘。

不能从最小版外推到所有代理、所有模态或完整 unlearning。

---

## 16. 给人的研究者检查点

服务器 AI 每完成一个阶段后，研究者只需检查：

1. `gate.json` 是否诚实反映结果；
2. 关键差异是否来自策略，而非模型、数据或 `top_k` 改动；
3. 测试集是否完全隔离；
4. 失败案例是否与指标方向一致；
5. 下一阶段是否仍然服务于 Q1-Q4，而不是无关的工程扩展。

建议在 STAGE 2、STAGE 5、STAGE 6 三个节点各做一次人工审阅。这三处分别决定“课题是否成立”“图感知是否成立”“延迟归因是否成立”。

---

## 17. 参考文件（本计划所依据的已提供材料）

1. `2502.12110_A-MEM-Agentic-Memory-for-LLM-Agents_中文译文.pdf`：第 3 节方法、第 4 节实验、附录 A/B。
2. `2606.25115_Forget-to-Improve(1).pdf`：第 III 节 score/govern、第 IV-V 节实现与评估，尤其是净价值密度、按字节预算和组件消融。
3. `2606.20954_Learning-What-Not-to-Forget(1).pdf`：第 3 节 LRE、表 1/3/4、附录 B-E，尤其是 query-blind scorer、分组防泄漏、自监督标签和 replay 评价。

本计划中的论文数字只用于选择实验设置和 sanity check；最终报告必须以本地运行结果为准。


