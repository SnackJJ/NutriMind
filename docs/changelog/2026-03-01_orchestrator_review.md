# Code Review: orchestrator.py

**日期**: 2026-03-01
**文件**: `src/orchestrator/orchestrator.py`
**审查者**: Claude Code

## 修复内容

### 1. 裸 `except:` 子句 (严重)

**问题**: 第45行和第59行使用 `except:` 捕获所有异常，会吞掉 `KeyboardInterrupt` 等系统异常，且无法追踪错误原因。

**修复**:
```python
# 之前
except:
    return OrchestratorConfig()

# 之后
except FileNotFoundError:
    logger.debug(f"Config file not found at {config_path}, using defaults")
    return OrchestratorConfig()
except Exception as e:
    logger.warning(f"Failed to load orchestrator config: {e}, using defaults")
    return OrchestratorConfig()
```

### 2. JSON 注入漏洞 (严重)

**问题**: `format_error_response` 使用 f-string 直接拼接错误消息，如果消息包含双引号会破坏 JSON 格式。

**修复**:
```python
# 之前
return f"<tool_response>\n{{\"status\": \"error\", \"error_type\": \"{error.type}\", \"message\": \"{error.message}\"}}\n</tool_response>"

# 之后
error_dict = {"status": "error", "error_type": error.type, "message": error.message}
return f"<tool_response>\n{json.dumps(error_dict, indent=2)}\n</tool_response>"
```

### 3. 未使用的参数

**问题**: `orchestrate(user_input, user_context=None)` 中 `user_context` 从未使用。

**修复**: 移除该参数。

### 4. 硬编码配置路径

**问题**: `"configs/orchestrator.yaml"` 是相对路径，从不同工作目录运行会失败。

**修复**:
```python
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
```

### 5. 缺少类型注解

**问题**: `execute_tool` 函数缺少返回类型。

**修复**: 添加 `-> Dict[str, Any]`

### 6. 工具映射重复创建

**问题**: 每次调用 `execute_tool` 都重建工具字典。

**修复**: 提取为模块级常量：
```python
TOOL_REGISTRY: Dict[str, Callable] = {
    "search_food": search_food,
    # ...
}
NO_ARGS_TOOLS = frozenset(["get_today_summary"])
```

### 7. 缺少输入验证

**问题**: `user_input` 无长度限制或空值检查。

**修复**:
```python
MAX_INPUT_LENGTH = 4096

def orchestrate(user_input: str) -> str:
    if not user_input or not user_input.strip():
        return "Error: Empty input provided."
    if len(user_input) > MAX_INPUT_LENGTH:
        return f"Error: Input exceeds maximum length of {MAX_INPUT_LENGTH} characters."
```

### 8. 异常被意外捕获

**问题**: `ToolExecutionError` 会被后续的 `except Exception` 捕获并重新包装。

**修复**:
```python
except ToolExecutionError:
    raise
except Exception as e:
    raise ToolExecutionError("internal_error", str(e))
```

## 最佳实践总结

| 模式 | 说明 |
|------|------|
| 异常分层捕获 | 先捕获具体异常，再捕获通用异常 |
| `json.dumps()` | 构建 JSON 字符串时始终使用，避免注入 |
| `Path(__file__)` | 构建相对于源文件的路径 |
| `frozenset` | 不可变集合用于常量 |
| 输入验证 | 在系统边界验证长度和空值 |
