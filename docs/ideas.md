# Ideas & Future Directions

> 个人笔记，不提交到 git

## 2024-02 讨论记录

### 1. Orchestrator 命名思考

当前 `orchestrator` 模块实际上是 **Single Agent Loop** (ReAct pattern)，而非 Multi-Agent Orchestrator。

更准确的命名选择：
- `AgentLoop`
- `AgentExecutor` (LangChain 风格)
- `ReActRunner`

**结论**: 保留现有命名，功能上确实在"编排"推理、工具执行、状态转换。

---

### 2. Multi-Agent 扩展方向

#### 可能的架构

```
┌────────────────────────────────────────────────────────┐
│              Coordinator Agent (3B)                     │
│         理解用户意图 → 分解任务 → 调度 → 整合            │
└────────────┬──────────────┬──────────────┬─────────────┘
             │              │              │
             ▼              ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Lookup Agent │ │ Advisor Agent│ │ Tracker Agent│
    │   查询食物    │ │  给出建议     │ │  追踪记录    │
    └──────────────┘ └──────────────┘ └──────────────┘
```

#### 实现方式

| 方案 | 描述 |
|------|------|
| 同模型多角色 | 一个 3B，不同 system prompt 扮演不同 agent |
| 3B + 小模型 | Coordinator 3B，Workers 用 0.5B/1B |
| 3B + Critic | Actor-Critic 架构，Critic 生成 reward signal |

#### 结论

**训练层 vs 应用层分离**：
- 训练：专注训练一个好的 actor 模型
- 应用：multi-agent 是部署时的编排选择，用同一个模型即可

---

### 3. RL 训练核心设计

#### Reward 组成

```
R_total = α·R_outcome + β·R_process + γ·R_efficiency

R_outcome (结果奖励):
├── 最终答案正确性
└── 用户满意度代理指标

R_process (过程奖励) ← 关键创新点:
├── 工具选择正确性 (每一步)
├── 参数格式正确性
├── 推理链连贯性 (<think> 质量)
└── 数据依赖正确性

R_efficiency (效率奖励):
├── 工具调用次数惩罚
└── 避免冗余调用
```

#### 消融实验设计

| 实验 | 配置 | 验证什么 |
|------|------|---------|
| Exp 1 | R = R_outcome only | Sparse reward baseline |
| Exp 2 | R = R_outcome + R_process | Process reward 帮助 |
| Exp 3 | 完整 R | 效率惩罚影响 |
| Exp 4 | 不同权重 | 敏感性分析 |

---

### 4. 未来可能的扩展

#### 优先级排序

1. **当前重点**: 完成 SFT + GRPO 训练，产出可用模型
2. **后续实验**: Single vs Multi-Agent 对比 (应用层)
3. **进阶方向**: 训练 Critic Model，替代规则 reward

#### 潜在研究问题

- 小模型 (3B) 场景下，什么样的 reward 设计最有效？
- Process Reward 对 agentic 能力的提升有多大？
- Multi-agent 编排对小模型是否有收益？

---

### 5. 技术债务 / 待优化

- [ ] 考虑将 `orchestrator/` 重命名为 `agent/`
- [ ] 分批提交 git commits (更原子化)
- [ ] unsloth 依赖管理优化

---

*Last updated: 2024-02*
