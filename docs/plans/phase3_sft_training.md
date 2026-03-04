# Phase 3: SFT 训练

> 优先级: Foundation | 预估工时: 2-3 天 | 依赖: Phase 2

## 🎯 目标

在 Qwen2.5-3B-Instruct 上进行 SFT，建立模型的 function calling 格式、工具选择和基本 agentic 能力基线。

## 📋 交付物

- [x] SFT 训练脚本 (`training/sft/train.py`)
- [ ] SFT 后模型 checkpoint
- [ ] 基线评估报告 (T1 ≥ 95%, Format ≥ 98%)
- [ ] 训练日志与曲线 (Wandb)

## 📝 详细任务

### Task 3.1: 训练脚本准备 (0.5 天) ✅

- [x] **3.1.1** 实现 `training/sft/train.py`
  - 加载 Qwen2.5-3B-Instruct (bfloat16)
  - 配置 LoRA (r=16, alpha=32, target: q/k/v/o_proj)
  - 配置 SFTTrainer
  ```python
  TrainingArguments(
      output_dir="./models/nutrimind-3b-sft",
      num_train_epochs=3,
      per_device_train_batch_size=4,
      gradient_accumulation_steps=4,
      learning_rate=2e-5,
      warmup_ratio=0.1,
      logging_steps=10,
      save_strategy="epoch",
      bf16=True,
      gradient_checkpointing=True,
  )
  ```

- [x] **3.1.2** 实现数据加载
  - 从 `training/sft/data/train.jsonl` 加载
  - 转换为 chat format (与 Qwen2.5 tokenizer 适配)
  - `max_seq_length=2048`, `packing=True`

- [x] **3.1.3** 配置 Wandb 追踪
  - Project: `nutrimind-sft`
  - 记录: loss, learning rate, gradient norm

### Task 3.2: 训练执行 (1 天)

- [ ] **3.2.1** GPU 服务器环境配置
  - 确认 GPU (RTX 4090 或等效)
  - 安装依赖，同步代码与数据
  - tmux session 保活

- [ ] **3.2.2** 启动训练
  - 预估: ~3000 样本 × 3 epoch × seq_len 2048
  - RTX 4090 (24GB): 预估 2-4 小时
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
  merged_model.save_pretrained("models/nutrimind-3b-sft-merged")
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
  - 对比 base model (Qwen2.5-3B-Instruct) 作为参照
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
