# Phase 3: SFT 训练

> 优先级: Foundation | 预估工时: 2-3 天 | 依赖: Phase 2.6 (Trajectory Collection)

## 🎯 目标

在 Qwen3-4B 上进行 SFT，建立模型的 function calling 格式、工具选择和基本 agentic 能力基线。

## 📊 数据概况 (2026-03-22 实测)

| 统计项 | 值 |
|--------|-----|
| 训练样本数 | 1,449 |
| 总 token 数 (1 epoch) | 4,172,166 |
| 总 token 数 (3 epochs) | 12,516,498 |
| 平均 token/样本 | 2,879 |
| P95 token 长度 | 5,342 |
| 超 8192 样本数 | 11 (0.8%, 截断) |

**Tier 分布**: T0=78, T1=413, T2=319, T3=282, T4=199, error-recovery=158

## 📋 交付物

- [x] SFT 训练脚本 (`src/training/sft/train.py`)
- [x] Adapter 合并脚本 (`src/training/sft/merge_adapter.py`)
- [x] 评估脚本 (`src/training/sft/evaluate.py`)
- [ ] SFT 后模型 checkpoint
- [ ] 基线评估报告 (T1 ≥ 95%, Format ≥ 98%)
- [ ] 训练日志与曲线 (Wandb)

## 📝 详细任务

### Task 3.1: 训练脚本准备 (0.5 天) ✅

- [x] **3.1.1** 实现 `training/sft/train.py`
  - 加载 Qwen3-4B (4bit, Unsloth)
  - 配置 LoRA (r=16, alpha=16, target: q/k/v/o/gate/up/down_proj)
  - 配置 SFTTrainer (loss masking: assistant turns only)
  ```python
  TrainingArguments(
      output_dir="./models/nutrimind-4b-sft",
      num_train_epochs=3,
      per_device_train_batch_size=1,
      gradient_accumulation_steps=16,
      learning_rate=2e-5,
      warmup_ratio=0.1,
      logging_steps=10,
      save_strategy="epoch",
      bf16=True,
      optim="adamw_8bit",
  )
  ```

- [x] **3.1.2** 实现数据加载
  - 从 `data/trajectories/sft_train_trajectory.jsonl` 加载
  - 转换为 chat format (与 Qwen3 tokenizer 适配, `apply_chat_template(enable_thinking=True)`)
  - `max_seq_length=8192`, `packing=False`, `group_by_length=True`
  - Loss masking: `train_on_responses_only(instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")`

- [x] **3.1.3** 配置 Wandb 追踪
  - Project: `nutrimind-sft`
  - 记录: loss, learning rate, gradient norm

### Task 3.2: 训练执行 (1 天)

- [ ] **3.2.1** GPU 服务器环境配置
  - 确认 GPU (RTX 4090 或等效)
  - 安装依赖，同步代码与数据
  - tmux session 保活

- [ ] **3.2.2** 启动训练
  - 实测: 1,449 样本 × 3 epoch = 4,347 样本
  - effective batch = 1 × 16 = 16 → ~272 steps total (~91 steps/epoch)
  - RTX 4090 (24GB): 预估 15-18 GB VRAM
  - 监控: GPU 利用率 > 90%, 显存使用合理

- [ ] **3.2.3** 中间检查
  - Epoch 1 结束: 检查 loss 曲线是否收敛
  - Epoch 2 结束: 快速评估 T1 格式准确率
  - 异常检查: loss spike, NaN, 显存溢出

### Task 3.3: Adapter 合并 (0.25 天) ✅ (脚本已就绪)

- [x] **3.3.1** 合并 LoRA adapter 到 base model
  ```python
  from peft import PeftModel
  model = PeftModel.from_pretrained(base_model, adapter_path)
  merged_model = model.merge_and_unload()
  merged_model.save_pretrained("models/nutrimind-4b-sft-merged")
  ```
  > 实现: `src/training/sft/merge_adapter.py`

- [x] **3.3.2** 验证合并后模型可正常推理
  > 实现: `merge_adapter.py --verify` 选项

### Task 3.4: SFT 基线评估 (0.5-1 天) ✅ (脚本已就绪)

- [x] **3.4.1** 设计评估数据集 (独立于训练数据)
  - T1 评估: 50 条单步工具调用
  - T2 评估: 30 条多步工具链
  - T3 评估: 20 条条件分支
  - T4 评估: 15 条安全边界声明（Type A/B/C，验证模型能精确识别高危场景并输出免责声明）
  - Format 评估: 混合 100 条
  - Pure QA: 10 条
  > 实现: `src/training/sft/generate_eval_data.py`

- [x] **3.4.2** 实现评估脚本 `training/sft/evaluate.py`
  - **Format Validity**: `<tool_call>` JSON 是否可解析
  - **Tool Selection Accuracy**: 是否选了正确的工具
  - **T1 Accuracy**: 单步任务端到端正确率
  - **Answer Quality**: LLM-as-judge (Qwen-Max)
  - 对比 base model (Qwen3-4B) 作为参照
  > 实现: `src/training/sft/evaluate.py`

- [ ] **3.4.3** 输出评估报告
  ```
  ┌──────────────────────────────────┐
  │     SFT Baseline Evaluation     │
  ├───────────────┬────────┬────────┤
  │ Metric        │ Target │ Actual │
  ├───────────────┼────────┼────────┤
  │ T1 Accuracy   │ ≥ 95%  │        │
  │ Format Valid  │ ≥ 98%  │        │
  │ T2 Accuracy   │ N/A    │        │
  │ T3 Accuracy   │ N/A    │        │
  │ T4 Safety     │ N/A    │        │
  │ QA Quality    │ ≤ -3%  │        │
  └───────────────┴────────┴────────┘
  ```

### Task 3.5: 问题诊断与迭代 (预留 0.5 天)

- [ ] **3.5.1** 如果 T1 < 95% 或 Format < 98%
  - 分析失败案例
  - 检查数据质量问题
  - 考虑: 增加 epoch / 调整 lr / 增加 T1 数据比例
  - 重新训练 (快速迭代)

- [ ] **3.5.2** 如果 QA 质量严重退化
  - 检查 Pure QA 数据比例是否充足
  - 考虑增加通用对话数据

## ✅ Phase 3 完成标准

| 检查项 | 标准 |
|--------|------|
| T1 单步工具调用准确率 | ≥ 95% |
| 工具调用格式有效性 | ≥ 98% |
| QA 质量退化 | ≤ 3% vs base model |
| 模型 checkpoint | 已保存 + adapter 已合并 |
| 评估报告 | 完整输出 + 失败案例分析 |
| Wandb 日志 | 训练曲线正常收敛 |
