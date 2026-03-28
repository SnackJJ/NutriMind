# GRPO env_state 设计重构

**日期**: 2026-03-28
**模块**: docs/plans/phase4_grpo.md
**类型**: 设计重构

## 变更内容

### §4.2.2 Metadata per prompt

**Before**: `env_state` 嵌入在 prompt metadata 中
```json
{
    "query": "...",
    "tier": "T3",
    "env_state": { ... }  // 耦合
}
```

**After**: `env_state` 由环境层管理，prompt 只保留 `query` + `tier`
```json
{
    "query": "...",
    "tier": "T3"
}
```

### 新增 §4.2.2a Environment State Design

1. **工具依赖分类**:
   - 静态工具 (`get_food_nutrition`, `retrieve_knowledge`): 不依赖 env_state
   - 读取工具 (`get_today_summary`, `get_history`): 依赖 env_state
   - 写入工具 (`log_meal`, `set_goal`): 依赖 env_state + 有副作用

2. **env_state 结构**:
   - `user_id`, `user_profile`, `user_goals`
   - `meals_today` (当天记录)
   - `meal_history` (历史数据)

3. **Group-level tool cache**:
   - 确保同一 group 的 G 个 rollout 中，相同 tool 调用返回相同结果
   - 缓存 key: `(group_id, tool_name, params)`

### §4.4.2 Tool Determinism

简化为引用 §4.2.2a 的设计，删除重复的 `DeterministicRolloutEnvironment` 代码。

## 设计理由

1. **解耦**: prompt 数据与环境配置分离
2. **一致性**: group-level 缓存确保 GRPO 优势估计的公平性
3. **灵活性**: env_state 可从 prompt 内容推断生成，不需要手动标注
