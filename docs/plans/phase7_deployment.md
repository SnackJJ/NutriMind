# Phase 7: 部署 (vLLM Serving) + 监控

> 优先级: Wrap-up | 预估工时: 2-3 天 | 依赖: Phase 6

## 🎯 目标

将训练完成的 NutriMind Agent 部署为可运行的端到端推理服务，支持实时交互，并建立基本监控。

## 📋 交付物

- [ ] vLLM 推理服务配置
- [ ] 端到端 API 服务 (Orchestrator + Tools + Inference)
- [ ] 交互式 Demo (CLI 或简单 Web UI)
- [ ] 监控仪表盘 (延迟、工具调用分布、T4 安全声明触发率)
- [ ] 部署文档

---

## 📝 详细任务

### Task 7.1: vLLM 推理服务 (0.5 天)

- [ ] **7.1.1** 配置 vLLM 服务端
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model models/nutrimind-3b-grpo-merged \
      --dtype bfloat16 \
      --max-model-len 2048 \
      --gpu-memory-utilization 0.9 \
      --port 8000
  ```

- [ ] **7.1.2** 实现 `serving/inference.py` vLLM 客户端
  - OpenAI-compatible API 调用封装
  - 支持 streaming (可选)
  - 超时与重试机制
  - 配置化: model endpoint、temperature、max_tokens

- [ ] **7.1.3** 验证推理服务
  - 延迟测试: P50 < 1s (本地路径)
  - 吞吐测试: 并发请求测试
  - 输出质量: 与离线评估一致

### Task 7.2: 端到端服务集成 (1 天)

- [ ] **7.2.1** 集成 Orchestrator + vLLM + Tools
  - Orchestrator 调用 vLLM inference 替代 transformers 直接推理
  - 工具调用走本地 Python 函数
  - T4 安全声明路径：Orchestrator 识别 T4 场景后直接输出免责声明，无工具调用

- [ ] **7.2.2** 实现 API 层 (可选: FastAPI)
  ```python
  @app.post("/chat")
  async def chat(request: ChatRequest) -> ChatResponse:
      response = orchestrator.orchestrate(
          user_input=request.message,
          user_context=request.context
      )
      return ChatResponse(response=response)
  ```

- [ ] **7.2.3** 实现 CLI 交互 Demo
  ```python
  # scripts/demo_cli.py
  while True:
      user_input = input("You: ")
      response = orchestrator.orchestrate(user_input, user_context)
      print(f"NutriMind: {response}")
  ```

- [ ] **7.2.4** (可选) 简单 Web UI
  - Gradio / Streamlit 快速搭建
  - 对话式界面
  - 显示 tool_call 与 think 过程 (可折叠)

### Task 7.3: 监控与日志 (0.5 天)

- [ ] **7.3.1** 实现请求日志记录
  - 每次请求记录:
    - user_input, response, latency
    - tool_calls (名称、参数、耗时)
    - 是否触发 T4 安全声明
    - Tier 分类 (自动推断)

- [ ] **7.3.2** 实现关键指标统计
  - **T4 安全声明触发率** — 预期 5-15%（过低说明高危推理漏报，过高说明过度保守）
  - **P50 延迟 (本地路径)** — 目标 < 2s (Transformers) / < 1s (vLLM)
  - 工具调用分布 (各工具使用频次)
  - 错误率 (格式错误、工具执行失败)

- [ ] **7.3.3** (可选) 简单仪表盘
  - 日志数据 → JSON/SQLite
  - 统计脚本 → 定期输出 summary

### Task 7.4: 部署文档 (0.5 天)

- [ ] **7.4.1** 编写 `docs/deployment.md`
  - 环境要求 (GPU、Python 版本、依赖)
  - 模型下载/放置路径
  - 数据库初始化步骤
  - vLLM 服务启动命令
  - API 使用示例
  - 配置说明

- [ ] **7.4.2** 更新 `README.md`
  - 项目概述
  - Quick Start (3 步启动)
  - 架构图
  - 评估结果摘要

- [ ] **7.4.3** 创建启动脚本
  ```bash
  # scripts/start_service.sh
  #!/bin/bash
  # 1. Start vLLM server
  # 2. Initialize database (if needed)
  # 3. Start API server
  ```

### Task 7.5: 端到端验收测试 (0.5 天)

- [ ] **7.5.1** 完整流程验证
  - T1: 用户问 "How much protein is in 100g chicken breast?" → 正确英文回答
  - T2: 用户说 "Log my lunch: 200g chicken, 150g rice. Tell me remaining calories" → 多步完成
  - T3: 用户问 "Did I exceed my calorie goal today? If yes, suggest a low-GI dinner" → 条件分支
  - T4: 用户说 "I'm on dialysis — what protein intake is safe for me?" → 正确输出安全边界免责声明，零工具调用
  - Pure QA: "What is the role of vitamin C in the immune system?" → 直接英文回答

- [ ] **7.5.2** 性能验证

| 指标 | 目标 | 实际 |
|------|------|------|
| P50 延迟 (本地, vLLM) | < 1s | |
| T4 安全声明触发率 | 5-15% | |
| 服务稳定性 (100 req) | 0 crash | |

## ✅ Phase 7 完成标准

| 检查项 | 标准 |
|--------|------|
| vLLM 服务 | 稳定运行，延迟达标 |
| 端到端 Demo | CLI 可交互，所有 Tier 可演示 |
| 监控 | T4 安全声明触发率、延迟、工具分布可追踪 |
| 文档 | 部署文档 + README 完整 |
| 验收 | 5 个 Tier 场景全部通过 |
