# NutriMind Agent — 架构思路对齐文档

> 写于：2026-03-01  
> 目的：对齐训练策略、应用设计、工具设计、数据采集各环节的核心思路，避免环节脱节

---

## 一、项目本质：这是一个什么项目？

**目标**：把一个通用 3B 小模型（Qwen2.5-3B-Instruct）通过 SFT + GRPO，训练成一个**专用营养领域 Agent**，部署在个人设备上运行。

### 核心定位
- **不是**通用 Agent，**是**针对 NutriMind 应用环境特化的专用模型
- **不是**大模型替代品，**是**在固定工具集内做出最优决策的小模型
- 类比：AlphaGo 之于围棋——不玩象棋，但围棋能力超越通用棋手

### 这意味着
- 泛化性弱是预期结果，不是缺陷
- 训练数据必须与部署环境精确对齐
- 环境（工具集、prompt 格式、reward 定义）必须在训练前冻结

---

## 二、整体架构：三个层次

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1：用户层（User-facing）                                     │
│                                                                   │
│  用户发消息                                                         │
│     ↓                                                             │
│  chat.py — 多轮 session 管理 + user_profile.md 注入 system prompt  │
└────────────────────────────┬─────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────────┐
│  Layer 2：Orchestrator 层（推理控制）                               │
│                                                                   │
│  Orchestrator — 状态机：                                           │
│    START → INFERENCE → TOOL_CALL → TOOL_EXEC → CHECK_LIMIT        │
│                ↓              ↓                   ↓               │
│             ANSWER          ERROR              dead loop 检测      │
│                                                                   │
│  InferenceBackend（可切换）：                                       │
│    - TransformersBackend：本地 transformers 加载 3B 模型            │
│    - VLLMBackend：vLLM 高性能推理 server                           │
│    - MockBackend：测试用                                           │
└────────────────────────────┬─────────────────────────────────────┘
                             ↓ 模型输出：<think>...</think> + <tool_call>{...}</tool_call>
┌──────────────────────────────────────────────────────────────────┐
│  Layer 3：工具层（Tool Execution）                                  │
│                                                                   │
│  6 个原子工具（接口已冻结）：                                           │
│  search_food / calculate_meal                                       │
│  log_meal / get_today_summary / get_history                         │
│  retrieve_knowledge                                                 │
│                                                                   │
│  数据持久化：                                                        │
│  - data/usda.db（USDA 营养数据库，23.7MB）                          │
│  - data/user.db（用户饮食记录，SQLite）                              │
│  - data/user_profile.md（用户档案，人类可读可编辑）                  │
│  - data/chroma/（RAG 向量数据库）                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、工具设计：6 个原子工具（已重构）

> **2026-03-01 设计决策**：依据工具原子化原则（单一职责，类比 Claude Code 的 Read/Write/Edit 设计），
> 将原 5 工具重构为 6 个原子工具。核心变化：
> 1. 删除 `call_expert_nutritionist`（见第四节 T4 重定义）
> 2. 拆解 `log_user_data` God Tool → 3 个原子工具
> 3. `search_food` 与 `calculate_meal` 完全解耦
> 4. `retrieve_knowledge` 新增 `domain` 参数

### 工具接口（已冻结）

| 工具 | 职责 | 读/写 | 使用场景 |
|------|------|-------|----------|
| `search_food(name)` | 查单种食物营养（USDA，per 100g） | 只读 | T1 |
| `calculate_meal(foods[{name, amount_grams}])` | 多食物汇总营养，完全独立实现 | 只读 | T2 |
| `log_meal(meal_type, foods[{name, amount_grams}])` | 记录一餐到 user.db | **写入** | T2 |
| `get_today_summary()` | 今日摄入总量 + 剩余热量预算 | 只读 | T2-T3 |
| `get_history(days, metric)` | 多日营养趋势查询 | 只读 | T3 |
| `retrieve_knowledge(query, top_k, domain)` | RAG 检索，可按领域过滤 | 只读 | T3 |

### 工具设计细节

**`search_food` 与 `calculate_meal` 完全解耦**：
- **旧设计（有问题）**：`calculate_meal` 内部调用 `search_food` 逻辑，导致训练轨迹冗余
  （T2 样本：`search_food × n` → `calculate_meal`，等于查了两遍）
- **新设计（解耦）**：两工具各自独立实现数据库查询
  - `search_food` = 纯查找，返回食物完整营养信息（适合探索性查询）
  - `calculate_meal` = 精确计算，接收食物+克重列表，返回汇总宏量（适合记录一餐）
- **训练轨迹 canonical pattern**：
  - T1：`search_food("鸡胸肉")` → 直接回答
  - T2（记录）：`calculate_meal([{name: "鸡胸肉", amount_grams: 200}, ...])`（**一步完成，不再需要先 search**）

**`retrieve_knowledge` 新增 `domain` 参数**：
```
domain 可选值：
  "medical"     — 疾病营养（糖尿病/高血压/痛风等）
  "dietary"     — 通用饮食指南（DGA）
  "sports"      — 运动营养
  "supplement"  — 营养补充剂
  null（默认）  — 全库检索
```
模型学会在医疗场景传 `domain="medical"`，提升检索精准度。

**`get_profile` 为什么被删除**：
`user_profile.md` 在每次对话开始时已注入 system prompt，模型一开始就掌握用户基本信息，
无需通过工具读取。`update_profile` 移出 agent 工具集，改为应用层单独管理（不参与训练）。

### 用户数据持久化：两层设计（不变）

```
user_profile.md  ← 静态配置（Markdown，人类可读可编辑，注入 system prompt）
  包含：年龄 / 身高 / 体重 / TDEE / 目标 / 活动水平 / 饮食限制 / 健康状况
  意义：每次对话自动携带，模型无需调用工具就知道用户基本情况
  注：新增 TDEE 字段（每日总能量消耗），避免模型现场计算引入误差

data/user.db     ← 动态记录（SQLite，结构化聚合查询）
  包含：meal_logs 表（每餐记录）+ daily_summary 视图（每日汇总）
  意义：支持"今日剩余热量""最近7天蛋白质趋势"等聚合查询
  为什么不用 Markdown：聚合查询（SUM/AVG/WHERE）依赖 SQL，
  让 3B 模型自己解析 Markdown 做求和极不可靠
```

---

## 四、任务分级（Tier）：训练边界的核心

> **2026-03-01 重大更新**：T4 定义从「升级到大模型」改为「安全边界声明」。
> 原因：call_expert 依赖云端 API，与本地部署架构矛盾；
> 且 reward hacking 风险高（模型倾向于将所有问题都升级以规避重罚）。
> Multi-agent 方案经分析不适合 3B 小模型（知识边界问题无法通过 prompt 注入解决）。

模型需要学会的最重要判断是：**这个任务属于哪个 Tier，我应该调用哪些工具？**

```
T1 — 单步查询（search_food × 1）
  "100g 鸡胸肉多少蛋白质？"
  工具链：search_food("鸡胸肉") → 直接回答

T2 — 多步工具链（calculate_meal / log_meal / get_today_summary）
  "我午饭吃了鸡胸肉+米饭，帮我记录并告诉我今天还能吃多少"
  工具链：calculate_meal([...]) → log_meal([...]) → get_today_summary() → 回答
  注：T2 不再需要 search_food × n，calculate_meal 已完全独立

T3 — 条件推理 + 多步规划（get_today_summary + retrieve_knowledge 组合）
  "我今天是否超出热量目标？如果超了推荐低GI晚餐"
  工具链：get_today_summary() → retrieve_knowledge("低GI晚餐", domain="dietary") → 综合回答

  T3 强化 planning（enhanced multi-step planning）：
  "我有糖尿病，今天吃了什么，还能吃什么晚饭"
  工具链：
    ① get_today_summary()            → 今日摄入情况
    ② retrieve_knowledge("糖尿病低GI晚餐", domain="medical") → 约束条件
    ③ search_food("豆腐")            → 候选食物验证
    ④ 综合三步结果，给出具体建议

T4 — 安全边界声明（无工具调用）⚠️ 重新定义
  触发条件：识别高危医疗关键词
    → 透析 / 外科手术后 / 活跃期癌症 / 器官移植 / 严重药物交互
  行为：输出标准免责声明，不调用任何工具，建议就医
  示例输出："您提到的[透析/CKD]情况涉及复杂的医疗营养管理，
            超出我的安全服务范围。请务必咨询您的主治医生
            或注册临床营养师获取个性化建议。"
  训练目标：模型学会"精准识别高危场景 → 直接输出声明，不调工具"

pure_qa — 无工具直接回答
  "什么是饱和脂肪？" / "每天应该喝多少水？"
```

### T4 边界训练的关键：三类样本（更新后）

```
Type A：看似复杂但 T3 能处理（训练不触发安全声明）
  "我有点高血压，应该少吃什么？"
  → retrieve_knowledge(domain="medical")，正常回答，不触发 T4

Type B：看似简单但必须触发安全声明（训练正确识别）
  "我在做透析，蛋白质吃多了有没有问题？"
  → 输出安全边界声明，不尝试用 retrieve_knowledge 作答

Type C：对话中途揭示高危条件（动态触发）
  "我想增加蛋白质，每天150g可以吗？（顺便说下我有CKD 3期）"
  → 看到 CKD 3期 后，立刻切换到安全声明模式

关键区分（T3 vs T4）：
  T3 边界：有医疗背景，但问题属于一般营养建议范畴，知识库能安全覆盖
  T4 边界：需要基于个体医疗参数做定量建议（剂量/用量/方案），超过通用知识边界
```

---

## 五、对话设计：多轮 + 跨 Session 持久记忆

### 多轮对话（in-session）
- `chat.py` 维护 `conversation_history` 列表
- 每次请求把完整历史 + system prompt（含 user_profile.md）传给 Orchestrator
- Orchestrator 只处理"这一轮"，不感知 session 概念

### 跨 Session 持久记忆（靠工具，不靠上下文窗口）
- 用餐记录 → `log_meal(...)` 写 user.db
- 查今日 → `get_today_summary()` 读 user.db
- 查历史 → `get_history(days, metric)` 读 user.db
- 用户档案 → `user_profile.md` 每次注入 system prompt

```
Session A（昨天）：      Session B（今天）：
  用户：记录我的晚饭        用户：今天还能吃多少热量？
  → log_meal 写 DB          → get_today 读 DB → 结合昨日记录回答
```

**结论**：多轮感知靠 history，跨 session 感知靠工具读取持久数据，两者职责不重叠。

---

## 六、Orchestrator：部署时的运行时核心

**Orchestrator 是部署态的中枢，不能绕过。**

### 它做什么
```
1. 接收 user_input + user_context
2. 调 InferenceBackend.generate() 让本地 3B 模型推理
3. 解析模型输出的纯文本：<think>...</think> + <tool_call>{...}</tool_call>
4. 调 execute_tool() 执行真实工具
5. 把 <tool_response> 注入回 messages，继续循环
6. 检测死循环、处理格式错误（parse_error → FORMAT_RECOVERY_PROMPT）
7. 超出轮次限制 → FORCE_ANSWER_PROMPT，强制给出最终答案
```

### 关键设计
- **文本格式**：本地 3B 模型用纯文本输出工具调用（`<tool_call>{json}</tool_call>`），不是 OpenAI function calling 协议
- **状态机**：显式状态转换，方便调试和监控
- **安全检查**：极端热量值（<800 或 >5000 kcal）触发硬约束，返回 0 reward；过敏原检测在 reward function 层实现

---

## 七、两个阶段的架构分离

### 训练阶段（Teacher 模型驱动数据采集）

```
pilot.py
  ↓ 调 Qwen API（教师模型，云端）
  ↓ 使用 OpenAI function calling 协议（方便，稳定）
  ↓ 执行本地真实工具（search_food 等）
  ↓ ⚠️ 采集的格式是 OpenAI 结构化 tool_calls
  ↓ 需要格式转换 → <think>/<tool_call> 文本格式
  ↓
SFT 训练数据（<think>/<tool_call> 格式）
  ↓
SFT 训练 Qwen2.5-3B
  ↓
GRPO 强化训练
```

### 部署阶段（本地 3B 模型运行）

```
用户
  ↓
chat.py（session 管理 + profile 注入）
  ↓
Orchestrator（状态机推理控制）
  ↓ InferenceBackend（Transformers / vLLM）
  ↓ 本地 3B 模型输出 <think>/<tool_call> 文本
  ↓
parse_model_output() → execute_tool()
  ↓
真实工具结果注入 <tool_response>
  ↓
OrchestratorResult → chat.py → 用户
```

**关键约束**：训练数据的格式必须和 Orchestrator 期望的格式一致（`<think>/<tool_call>` 纯文本）。

---

## 八、GRPO Reward 设计

> **2026-03-01 新增**：本节为之前文档缺失的 P0 内容。
> GRPO 的核心是 reward function，必须在数据采集前明确，否则 SFT 和 GRPO 训练方向对不上。

### 8.1 Reward 三维度拆解

**维度 A：格式合规性（Format Reward）** — binary，最优先
```
① 输出包含合法的 <think>...</think> 块                 +0.1
② 工具调用格式合法（<tool_call>{valid JSON}</tool_call>） +0.1
③ JSON 参数类型正确（无多余字段，必填项齐全）            +0.1
④ 最终回答不为空                                     +0.1
格式分满分 0.4，各项独立计算
```

**维度 B：工具使用合理性（Tool Use Reward）** — 最核心的训练信号
```
正确情况（加分）：
  - 选对工具 + 参数语义正确                             +0.3
  - T4 场景正确触发安全声明（无工具调用）                 +0.5  ← 安全敏感，加权
  - 工具步骤数最优（无冗余调用）                         +0.2

错误情况（扣分）：
  - T4 场景未触发声明，尝试用工具作答（漏报）             -0.6  ← 安全红线，重罚
  - 非 T4 场景触发安全声明（过度保守）                   -0.3  ← 轻罚
  - 选错工具（如 search_food 替代 get_today_summary）    -0.2
  - 冗余工具调用（search_food × n 后再 calculate_meal） -0.1
  - domain 参数语义错误                                -0.1
```

> **T4 漏报重罚 vs 过度保守轻罚（-0.6 vs -0.3）的依据**：
> 漏报高危问题（给了错误的医疗建议）的真实安全风险远高于过度保守（未给建议）。
> 但比值不应超过 3:1，否则模型找到捷径（全部声明）。
> 用 Type A 样本（大量"看似复杂但不应触发 T4"的训练数据）来对冲过度保守倾向。

**维度 C：结果质量（Outcome Reward）** — 针对可验证的输出
```
数值型答案（T1/T2，与真实工具返回值比对）：
  - 热量误差 < 5%       +0.3
  - 热量误差 5%-15%     +0.1
  - 热量误差 > 15%      0

建议类答案（T3/pure_qa，Teacher 模型 LLM-as-Judge 评分）：
  - 评分 4-5/5          +0.3
  - 评分 3/5            +0.1
  - 评分 < 3/5          0

T4 安全声明：
  - 包含具体风险说明（非通用拒绝）    +0.2
  - 建议了具体就医方向               +0.1
```

### 8.2 总 Reward 公式

```
reward = Format_reward + ToolUse_reward + Outcome_reward
理论区间：[-0.6, 1.5]，实践期望区间：[0.3, 1.0]
```

### 8.3 训练数据 Tier 比例要求

| Tier | 推荐占比 | 原因 |
|------|---------|------|
| T1 | 30% | 基础能力，高频场景 |
| T2 | 30% | 工具链核心，需充足示例 |
| T3 | 25% | Multi-step planning，训练难点 |
| T4（安全声明） | 10% | 过多会导致模型过度保守 |
| pure_qa | 5% | 防止模型什么都调工具 |

> T4 训练数据中，Type A（不触发声明）: Type B（触发声明）≈ 3:1，
> 确保模型学会精确判断边界，而非简单地"有医疗词就声明"。

---

## 九、知识库设计

### USDA 数据库（search_food 的底层）
- 覆盖约 35 万种食物的营养成分
- 缺点：中餐食物覆盖弱（已用 `chinese_foods.md` 补充）

### RAG 知识库（retrieve_knowledge 的底层）
- 存储位置：Chroma 向量数据库（需配置 domain metadata 字段支持过滤）
- 当前知识库文件（`data/knowledge/`）：
  - `dietary_guidelines.md` — DGA 2020-2025 要点（domain: dietary）
  - `medical_nutrition.md` — 糖尿病/高血压/痛风/减脂/增肌饮食原则（domain: medical）
  - `gi_food_table.md` — 常见食物 GI 值表（domain: dietary）
  - `chinese_foods.md` — 中餐食物营养参考
  - `sports_nutrition.md` — 运动营养指南（domain: sports）
  - `supplement_interactions.md` — 营养补充剂相互作用（domain: supplement）

### 知识库设计原则
- **聚焦不泛化**：只收录营养/饮食相关内容，不做通用健康 RAG
- **质量 > 数量**：200 条精准内容优于 10000 条低质内容
- **每条文档标注 domain**：支持 `retrieve_knowledge` 的 domain 参数过滤

---

## 十、扩展性设计

### 现阶段（普通用户）
- 工具接口固定不变（6 个原子工具）
- 模型训练覆盖 T1-T4（安全声明）和 pure_qa

### 下阶段（专业用户：运动员/慢病患者）扩展路径

```
扩展方式              成本         是否需要重训模型
添加知识库文档         极低         ❌ 不需要（加文档并重新 index Chroma）
更新 user_profile.md   极低         ❌ 不需要（profile 自动注入 context）
T4 安全声明兜底        零           ❌ 不需要（高危问题直接拒绝，离线可用）
添加新工具             中等         ⚠️ 需要补充数百条 SFT 数据做 LoRA 微调
扩展专业领域的训练      较高         ✅ 需要 LoRA 微调（几百条数据，1-2小时）
```

**核心思想**：`user_profile.md` 字段预留扩展空间 + T4 安全声明作为高风险问题的统一处理方式。
与旧设计的区别：`call_expert` 兜底被替换为「安全声明兜底」，无需联网，架构完全自洽。

---

## 十一、当前遗留问题追踪

### ✅ 已解决：call_expert 与离线部署矛盾
- **原问题**：`call_expert_nutritionist` 依赖云端 API，与本地部署定位矛盾，且存在 reward hacking 风险
- **解决方案**（2026-03-01）：删除该工具。T4 重定义为安全边界声明，模型直接输出免责声明并建议就医。Multi-agent 方案分析后不适合 3B 小模型（知识边界问题无法通过 prompt 注入解决）

### ✅ 已解决：GRPO Reward 设计缺失
- **原问题**：文档未定义 reward function，SFT 与 GRPO 训练方向对不上
- **解决方案**（2026-03-01）：见第八节，三维度 reward 设计（格式 + 工具合理性 + 结果质量）

### ✅ 已解决：search_food / calculate_meal 调用模式模糊
- **原问题**：两工具内部耦合，训练轨迹冗余（T2 需要先 search × n 再 calculate）
- **解决方案**（2026-03-01）：完全解耦，T2 轨迹改为直接调 `calculate_meal`，canonical pattern 已明确

### ⚠️ 未解决：工具实现需要重构（数据采集前必须完成）
- **问题**：`log_user_data` God Tool 需要拆分为 3 个原子工具；`calculate_meal` 需要与 `search_food` 解耦
- **影响**：采集前必须完成，否则工具 schema 和实现不一致
- **涉及文件**：`src/tools/log_user_data.py` → 拆分为 `log_meal.py` / `get_today_summary.py` / `get_history.py`

### ⚠️ 未解决：格式转换风险（数据采集前必须修复）
- **问题**：`collect_trajectories.py` 使用 OpenAI function calling 采集，格式不直接是 `<think>/<tool_call>` 纯文本
- **风险**：转换 bug 会静默污染整个训练集（不报错但数据错误）
- **解决方案**：保存前加 format conversion 函数，并做 round-trip 验证（转换 → 解析 → 比对参数一致性）

### ⚠️ 未解决：chat.py 绕过了 Orchestrator
- **问题**：chat.py 自己实现推理循环，调用 Qwen API，而非包装 Orchestrator
- **影响**：不影响数据采集；部署时需要修正
- **解决方案**：部署前将 chat.py 重构为 Orchestrator + InferenceBackend 模式

---

## 十二、近期执行优先级（更新版）


```
P0（影响训练数据，必须先做）：

  【最高优先级：语言架构】
  ✅ 全英语决策确定（见末尾语言架构章节）
  ✅ SYSTEM_PROMPT 更新为全英文版本（SOTA 文档已更新）
  ⬜ 知识库文档改造为英文（替换为 ADA/DGA/NIH 等英文权威来源）
  ⬜ 训练采集 query 改为英文（全部种子 query 翻译为英文，或直接用英文写新 query）
  ⬜ validate_rules.py 加语言合规性检验（think 块不含中文，回答为英文）

  【工具与数据】
  ✅ T4 架构决策（删除 call_expert，改为安全声明）
  ✅ GRPO reward 函数设计（见第八节）
  ⬜ 工具实现重构：
       log_user_data → log_meal / get_today_summary / get_history（3个原子工具）
       calculate_meal 与 search_food 完全解耦
  ⬜ retrieve_knowledge 工具加 domain 参数支持（+ Chroma metadata 配置）
  ⬜ user_profile.md 新增 TDEE 字段
  ⬜ collect_trajectories.py 格式转换 + round-trip 验证

P1（采集阶段同步）：
  ⬜ 知识库文档标注 domain 元数据（配合 Chroma domain 过滤）
  ⬜ T4 安全声明样本补充到采集 query 列表（Type A/B/C 三类，比例 3:1）
  ⬜ 部署方案：chat.py 加轻量翻译包装（中文用户输入→英文→模型→英文输出→中文）
  ⬜ chat.py 重构为包装 Orchestrator（部署时需要，采集阶段不影响）

P2（采集完成后）：
  ⬜ Pilot 验证 10 条：T4 触发边界 + 语言合规性 + 工具 canonical pattern
  ⬜ 大批量采集 500 条 trajectory
       （Tier 比例：T1 30% / T2 30% / T3 25% / T4 10% / pure_qa 5%）
  ⬜ SFT 训练（Qwen2.5-3B-Instruct）
  ⬜ GRPO 训练（按第八节 reward 公式）
  ⬜ 对比评估（SFT baseline vs GRPO vs 大模型 teacher）
```

---

## 十三、补充设计：Token 预算、OOD 策略、知识库版本管理

### 13.1 Token 预算分析（2026-03-01）

Qwen2.5-3B-Instruct 上下文窗口：32K token。

**典型对话 token 构成：**

```
固定部分（每次对话）：
  系统指令 + 工具格式规范          ~400 token
  6 个工具的 JSON Schema           ~700 token
  user_profile.md 注入             ~300-600 token
  ─────────────────────────────
  小计                             ~1400-1700 token

动态部分（每轮约）：
  用户消息                         ~50-150 token
  模型 <think> + <tool_call>       ~150-350 token
  工具返回值                       ~100-400 token
  模型最终回答                     ~100-300 token
  单轮小计                         ~400-1200 token

典型 T3 任务（3 步工具链 + 5 轮历史）：
  固定 1600 + 动态约 4200 ≈ 5800 token  ← 远低于 32K 上限
```

**结论：日常使用不存在上下文溢出风险。** 但需要防范两个边界场景：

**保护措施 1：`retrieve_knowledge` 返回值截断**
- 单条知识片段 ≤ 300 token（工具层截断）
- `top_k` 默认值 = 3（返回 3 条，总计 ≤ 900 token）

**保护措施 2：多轮对话滑动窗口（`chat.py` 实现）**
```python
MAX_HISTORY_TURNS = 10  # 超出时删除最早轮次
# 保留：system prompt（必须）+ 最近 10 轮 + 当前轮
# 超出时：从最早的非 system 轮次开始删除
```

---

### 13.2 OOD 降级策略（2026-03-01）

**三类 OOD 场景及处理方式：**

**类型 A：食物不在 USDA 数据库（`food_not_found`）**
- 工具返回 `{"status": "not_found", "suggestions": [...]}` 时，模型学会：
  告知用户食物未找到 → 提供相似替代品建议 → 或请用户提供英文名称
- 训练数据应包含 `food_not_found` 错误恢复轨迹（SOTA 文档已规划）

**类型 B：口语/方言/中英混杂的查询风格**
- Qwen2.5 预训练数据覆盖广，影响较小
- USDA 搜索支持模糊匹配，`food_name` 无需严格标准化
- **不需要专门处理**

**类型 C：完全超出营养领域（写诗、股市等）**
- 在 system prompt 中明确边界声明
- 训练数据加入少量 OOD 拒绝样本（~20 条，占比 < 1%）：
  ```
  用户："帮我写首诗"
  模型："我是 NutriMind 营养助手，专注于饮食和营养健康方面的问题。
        您有关于饮食、营养或健康饮食规划的问题我很乐意帮助！"
  ```

---

### 13.3 知识库版本管理（2026-03-01）

**更新流程（无需重训模型）：**
```
1. 新增/修改知识文档（data/knowledge/*.md）
2. 同一领域新版本替换旧版本（不并存，避免矛盾内容）
3. 运行 python scripts/update_knowledge.py → 增量更新 Chroma index（约 10s）
4. 在 data/knowledge/CHANGELOG.md 记录变更
```

**文档质量标准：**
- 来源优先级：权威机构（ADA/中国膳食指南/WHO）> 学术论文 > 科普文章
- 每条片段需标注来源机构 + 发布年份（metadata 字段）
- 禁止抓取未经验证的网络内容
- 同一 domain 内，新指南的推荐值优先，旧版本归档删除

**Chroma metadata 结构（每条文档）：**
```json
{
  "domain": "medical",
  "source": "ADA Standards of Care 2024",
  "year": 2024,
  "topic": "diabetes_nutrition"
}
```

---

## 【已决策】语言架构方案选择

> **状态：已决策 ✅**（2026-03-01）
> **优先级：P0 最高**——影响训练数据格式、system prompt、知识库、验证规则，所有采集工作前必须完成

### 决策结论：全英语（Full English）架构

```
选择方案：全英语——training queries、<think>、tool 参数、RAG 知识库、最终回答全部英文
决策日期：2026-03-01
决策理由：
  1. 内部一致性最优——think→tool→知识库全链路无语言切换，3B 学习负担最小
  2. 英文 CoT 推理质量在多步规划上有研究支持优势（T3 场景尤为明显）
  3. 实验发现：Qwen face 中文用户输入时 think 语言难以强制控制（对比实验）
     → 解法是：训练数据本身的 user query 也使用英文（或先翻译再采集）
  4. 营养领域权威知识来源（ADA/WHO/DGA 指南）本身就是英文，零翻译损失
  5. USDA 数据库本身是英文，food_name 参数天然英文，不再需要工具层翻译
```

### 最终语言架构（确定版）

| 组件 | 语言 | 说明 |
|------|------|------|
| System Prompt | **英文** | 已更新（见 SOTA 文档） |
| 训练 user queries | **英文** | 采集时使用英文问题（或翻译中文种子） |
| `<think>` 块 | **英文** | 训练数据全英文 query → think 自然英文 |
| Tool 参数（food_name） | **英文** | 直接传英文，无需转换层 |
| Tool 参数（其他） | **英文** | action/domain 等本来就是英文枚举 |
| RAG 知识库 | **英文** | ⬜ 需改造（见下） |
| 最终回答 | **英文** | 训练数据中回答为英文 |

> **部署时的用户体验**：如果最终部署面向中文用户，在 chat.py 层加轻量翻译包装
> （中文输入 → 英文 → 模型 → 英文输出 → 中文）。
> 这是 chat.py 的应用层问题，不影响模型训练。

### 知识库改造任务（P0）

当前中文知识库文件需要替换为英文版本：

| 文件 | 改造方案 |
|------|---------|
| `dietary_guidelines.md` | 直接使用 DGA 2020-2025 英文原版内容 |
| `medical_nutrition.md` | 使用 ADA / AHA 英文指南内容替换 |
| `gi_food_table.md` | GI 数据表有国际英文版，直接替换 |
| `sports_nutrition.md` | 使用 ISSN / ACSM 英文运动营养指南 |
| `supplement_interactions.md` | 使用 NIH ODS 英文补充剂数据 |
| `chinese_foods.md` | 保留但内容改为英文（中餐食物的英文名称和营养说明） |

好处：英文原版权威来源质量更高，无翻译损耗。