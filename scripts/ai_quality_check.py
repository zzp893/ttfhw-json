#!/usr/bin/env python3
"""
TTFHW JSON Quality Gate — AI Semantic Analysis Script (DeepSeek v4-pro).

Uses the DeepSeek API (OpenAI-compatible) with reasoning/thinking enabled to
perform semantic checks on verification report JSON files that cannot be done
with deterministic rules alone.

Checks:
- Chinese status value consistency
- failure_reason quality assessment
- documentation_gaps specificity
- problems_encountered closure (problem→solution matching)
- Process timeline details coherence

Usage:
    export DEEPSEEK_API_KEY=sk-...
    python ai_quality_check.py reports/file1.json [reports/file2.json ...]

Output: JSON to stdout with structure:
    {"pass": bool, "files": {path: {"pass": bool, "issues": [...]}}}
"""

import json
import os
import re
import sys
import zlib
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Token injection / token-bomb protection constants
# ---------------------------------------------------------------------------

# Hard limit: reject files larger than this (1 MB)
MAX_FILE_SIZE_BYTES = 1_000_000

# Hard limit: reject any single string value longer than this (50K chars)
MAX_STRING_LENGTH = 50_000

# Hard limit: the focused content sent to the AI must not exceed this (~60K chars)
# DeepSeek v4-pro has a large context window, but we keep it tight per file
MAX_FOCUSED_CONTENT_CHARS = 60_000

# If any string value repeats the same substring to pad tokens
# (e.g., "aaaa...aaaa" repeated 5000 times), flag as suspicious
MAX_REPETITION_RATIO = 0.3  # if >30% of the content is one repeating pattern

# Minimum estimated token count that triggers a hard rejection
MAX_ESTIMATED_TOKENS = 30_000

# Suspicious prompt-injection patterns in text fields
PROMPT_INJECTION_PATTERNS = [
    (re.compile(r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|directives?)',
                re.IGNORECASE), "prompt override attempt"),
    (re.compile(r'you\s+are\s+(now|no\s+longer)\b.{0,50}\b(assistant|validator|checker)',
                re.IGNORECASE), "role hijack attempt"),
    (re.compile(r'system\s*(prompt|message|instruction)\s*(:|is|was|now)\s*["\']?',
                re.IGNORECASE), "system prompt injection"),
    (re.compile(r'do\s+not\s+(check|validate|analyze|report|flag)\b',
                re.IGNORECASE), "suppression instruction"),
    (re.compile(r'output\s+(only|just|exactly)\s*["\']?\{\}', re.IGNORECASE),
     "output suppression attempt"),
    (re.compile(r'\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>', re.IGNORECASE),
     "LLM instruction tags"),
    (re.compile(r'\]\(javascript:', re.IGNORECASE),
     "markdown XSS in content"),
]


# ---------------------------------------------------------------------------
# System prompt for DeepSeek
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一个 TTFHW 验证报告 JSON 质量检查专家。你的任务是分析 JSON 验证报告中的语义一致性问题。

## JSON 报告结构

每份报告有 8 个顶层 key：
1. metadata — 验证运行的身份和时间信息
2. machine_spec — 宿主机/容器硬件和镜像信息
3. document_reading_summary — 从仓库文档中提取的信息
4. execution_log — 逐条命令执行记录
5. process_timeline — 按语义阶段划分的验证过程时间线
6. final_results — 5 项汇总结果（static_analysis/devcontainer/build/ut/sample）
7. documentation_gaps — 文档缺失/不足的列表
8. problems_encountered — 遇到的问题与解决方案

## 语义检查项目

请逐项检查以下内容。对每个发现的问题，提供：
- severity: "warning"（严重/可疑）或 "notice"（建议性）
- path: JSON 路径，如 $.final_results.build.status
- message: 中文描述，具体说明问题
- suggestion: 修复建议

### 检查 1: 状态值一致性
- final_results.build/ut/sample 的 status 取值是否合理？
- 已知合理值：成功、不成功、超时失败、未执行
- 如果 status 为 "不成功" 但 failure_reason 为空或不具体，标记 warning

### 检查 2: failure_reason 质量
- failure_reason 是否具体描述了失败原因？还是泛泛的占位符？
- 好的例子："cmake configure 阶段缺少 OpenSSL 库，CMakeLists.txt 第25行 find_package(OpenSSL REQUIRED) 失败"
- 差的例子："构建失败"、"未知错误"、空字符串
- 过于简短的（<20 字符）描述标记 warning

### 检查 3: documentation_gaps 质量
- 是否具体、有分类、可操作？
- 空数组标记 notice（建议补充文档缺口）
- 过于笼统的描述（如"文档不完整"）标记 warning
- 好的例子："README 未列出 OpenSSL 依赖，仅在 CMakeLists.txt 中找到" 是有价值的

### 检查 4: problems_encountered 闭环
- 每个 problem 是否有对应的 solution？
- solution 是否真的解决了所描述的 problem？
- problem 是否关联到了具体的 execution_log 条目（通过时间戳）？
- 不匹配或不完整的闭环标记 warning

### 检查 5: process_timeline details 匹配
- details 子字段是否与 step 类型匹配？
- 例如：build_attempt 应有 concurrency/artifact/error 等，不应有 file/files_read
- document_reading 应有 file/files_read/sections_read
- 明显不匹配的标记 notice

### 检查 6: 其他语义问题
- 报告各部分之间是否有矛盾的描述？
- execution_log 中有失败记录但 final_results 却标记为 "成功"？
- 时间线有明显的逻辑跳跃？

## 输出格式

你必须**只输出**一个 JSON 对象，不要有任何 markdown 标记、代码块标记或额外文字。

输出格式：
{
  "summary": "一句话总结分析结果",
  "issues": [
    {
      "severity": "warning",
      "path": "$.final_results.build.status",
      "message": "build status 为 '不成功' 但 failure_reason 过于简短",
      "suggestion": "建议补充具体的失败原因，如：'cmake 阶段报错：Could not find OpenSSL'"
    }
  ]
}

如果没有发现问题，issues 为空数组 []。

严格注意：
- 只输出 JSON，不要有 ```json 或 ``` 包围
- 所有字符串使用中文
- 路径格式为 $.xxx.yyy.zzz
"""


# ---------------------------------------------------------------------------
# Token injection protection
# ---------------------------------------------------------------------------

def check_file_size(filepath: str) -> Optional[str]:
    """Check file is not too large. Returns error message if rejected."""
    size = os.path.getsize(filepath)
    if size > MAX_FILE_SIZE_BYTES:
        return (f"File too large: {size / 1_000_000:.1f}MB (limit: "
                f"{MAX_FILE_SIZE_BYTES / 1_000_000:.0f}MB). "
                f"Rejected for token-bomb protection.")
    return None


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~1 token per 3 chars for Chinese, 4 chars for English.
    This is a conservative heuristic for DeepSeek's tokenizer."""
    # Count CJK characters (they use more tokens per char)
    cjk_chars = len(re.findall(r'[一-鿿　-〿＀-￯]', text))
    other_chars = len(text) - cjk_chars
    # CJK: ~1.5 chars/token, Other: ~4 chars/token (conservative)
    return int(cjk_chars / 1.2 + other_chars / 3.5)


def check_string_repetition(s: str) -> Optional[str]:
    """Check if a string contains suspiciously repetitive content.
    Returns description if suspicious, None if OK."""
    if len(s) < 200:
        return None

    # Compress with zlib — if compression ratio is extreme, content is repetitive
    try:
        original = s.encode('utf-8')
        compressed = zlib.compress(original, level=1)
        ratio = len(compressed) / max(len(original), 1)
        if ratio < 0.05:  # >95% compression ratio = highly repetitive
            return (f"Highly repetitive content detected "
                    f"(compression ratio {ratio:.2%}, likely token padding)")
    except Exception:
        pass

    # Check for single-character repetition
    for ch in set(s[:50]):  # check chars from the beginning
        if s.count(ch) / len(s) > 0.9 and len(s) > 500:
            return (f"Single-character repetition pattern detected "
                    f"(char '{ch}' repeated {s.count(ch)}/{len(s)} times)")

    return None


def scan_for_prompt_injection(s: str, json_path: str) -> List[Dict]:
    """Scan a string value for prompt injection patterns.
    Returns list of security issues found."""
    issues = []
    for pattern, desc in PROMPT_INJECTION_PATTERNS:
        match = pattern.search(s)
        if match:
            ctx_start = max(0, match.start() - 20)
            ctx_end = min(len(s), match.end() + 20)
            context = s[ctx_start:ctx_end].replace('\n', ' ')
            issues.append({
                "severity": "error",
                "check": "security_prompt_injection",
                "path": json_path,
                "message": f"检测到 AI 提示词注入: {desc} — \"...{context}...\"",
                "suggestion": "此内容可能试图操纵 AI 分析结果，请移除可疑指令"
            })
    return issues


def sanitize_str(s: str, max_len: int = MAX_STRING_LENGTH) -> Tuple[str, bool]:
    """Truncate a string value if it exceeds the maximum length.
    Returns (sanitized_string, was_truncated)."""
    if not isinstance(s, str):
        return str(s), False
    if len(s) > max_len:
        return s[:max_len] + f"\n\n[... 已截断，原长度 {len(s)} 字符，超出 AI 分析限制 ...]", True
    return s, False


def build_safe_focused_view(data: dict, filepath: str) -> Tuple[dict, List[Dict]]:
    """Build a safe, focused subset of the JSON data for AI analysis.

    This function:
    1. Extracts only the relevant fields (ignoring raw logs)
    2. Truncates overly long string values
    3. Checks for prompt injection in all text fields
    4. Checks for token-bomb patterns (repetition, padding)
    5. Enforces a total content size limit

    Returns (focused_view, security_issues_found).
    """
    security_issues = []

    def _walk_and_protect(obj, path=""):
        """Recursively walk the data, truncating and scanning strings."""
        if isinstance(obj, str):
            # Scan for prompt injection in ALL strings
            sec = scan_for_prompt_injection(obj, f"$.{path}" if path else "$")
            security_issues.extend(sec)

            # Check for repetitive/token-padding content
            rep_issue = check_string_repetition(obj)
            if rep_issue:
                security_issues.append({
                    "severity": "error",
                    "check": "security_token_bomb",
                    "path": f"$.{path}" if path else "$",
                    "message": f"检测到 token 炸弹: {rep_issue}",
                    "suggestion": "此内容可能是为了消耗 AI token 而构造的膨胀数据"
                })

            # Truncate long strings
            sanitized, truncated = sanitize_str(obj)
            if truncated:
                security_issues.append({
                    "severity": "warning",
                    "check": "security_string_truncated",
                    "path": f"$.{path}" if path else "$",
                    "message": f"字符串过长 ({len(obj)} 字符)，已截断至 {MAX_STRING_LENGTH} 字符",
                    "suggestion": "请检查此字段是否被注入了异常大量的内容"
                })
            return sanitized

        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                child_path = f"{path}.{k}" if path else k
                result[k] = _walk_and_protect(v, child_path)
            return result

        elif isinstance(obj, list):
            # Limit arrays to prevent abuse
            if len(obj) > 500:
                security_issues.append({
                    "severity": "warning",
                    "check": "security_array_truncated",
                    "path": f"$.{path}" if path else "$",
                    "message": f"数组过长 ({len(obj)} 项)，仅保留前 500 项进行分析",
                    "suggestion": "请检查是否被注入了异常的数组项"
                })
                obj = obj[:500]
            result = []
            for i, v in enumerate(obj):
                child_path = f"{path}.{i}" if path else str(i)
                result.append(_walk_and_protect(v, child_path))
            return result

        else:
            return obj

    # Build the focused view (same structure as before)
    focused = {
        "metadata": data.get("metadata", {}),
        "machine_spec": {
            "host_machine": {
                k: data.get("machine_spec", {}).get("host_machine", {}).get(k)
                for k in ["docker_version", "architecture"]
            },
        },
        "final_results": data.get("final_results", {}),
        "documentation_gaps": data.get("documentation_gaps", []),
        "problems_encountered": data.get("problems_encountered", []),
        "process_timeline": [
            {
                "step": e.get("step"),
                "result": e.get("result"),
                "details": e.get("details", {}),
            }
            for e in data.get("process_timeline", [])
            if isinstance(e, dict)
        ],
        "execution_summary": {
            "total_entries": len(data.get("execution_log", [])),
            "failed_entries": [
                {"command": e.get("command", ""), "success": e.get("success"),
                 "error": e.get("error", ""), "returncode": e.get("returncode")}
                for e in data.get("execution_log", [])
                if isinstance(e, dict) and not e.get("success", True)
            ],
        },
    }

    # Walk the focused view and apply all protections
    protected = _walk_and_protect(focused)

    return protected, security_issues


# ---------------------------------------------------------------------------
# DeepSeek API call (via OpenAI SDK)
# ---------------------------------------------------------------------------

def call_deepseek(system: str, user: str, api_key: str,
                  model: str = "deepseek-v4-pro",
                  timeout: int = 180) -> Optional[dict]:
    """Call DeepSeek API (OpenAI-compatible) and return parsed JSON."""
    try:
        from openai import OpenAI
    except ImportError:
        return {
            "summary": "AI analysis skipped: openai package not installed",
            "issues": [{
                "severity": "notice",
                "path": "$",
                "message": "AI 语义分析不可用：openai 包未安装 (pip install openai)",
                "suggestion": "在 CI 环境中安装 openai 包以启用 AI 分析"
            }]
        }

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=timeout,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )

        # Extract token usage for cost monitoring
        # DeepSeek pricing (CNY per 1M tokens):
        #   Input  (cache miss): ¥3   Input  (cache hit): ¥0.025   Output: ¥6
        PRICE_INPUT_CACHE_MISS = 3.0
        PRICE_INPUT_CACHE_HIT  = 0.025
        PRICE_OUTPUT           = 6.0

        usage = {}
        cost_input = 0.0
        cost_output = 0.0
        cached_tokens = 0
        uncached_tokens = 0

        if hasattr(response, 'usage') and response.usage:
            prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
            completion_tokens = getattr(response.usage, 'completion_tokens', 0)
            total_tokens = getattr(response.usage, 'total_tokens', 0)

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            # Check for cache hit tokens (DeepSeek may report via prompt_tokens_details)
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)
            uncached_tokens = prompt_tokens - cached_tokens

            # Calculate cost
            cost_input = (uncached_tokens / 1_000_000) * PRICE_INPUT_CACHE_MISS + \
                         (cached_tokens / 1_000_000) * PRICE_INPUT_CACHE_HIT
            cost_output = (completion_tokens / 1_000_000) * PRICE_OUTPUT

            usage["cached_tokens"] = cached_tokens
            usage["cost_input"] = round(cost_input, 6)
            usage["cost_output"] = round(cost_output, 6)
            usage["cost_total"] = round(cost_input + cost_output, 6)

            # DeepSeek thinking model: reasoning tokens
            if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                usage["reasoning_tokens"] = getattr(
                    response.usage.completion_tokens_details, 'reasoning_tokens', 0)

        text = response.choices[0].message.content or ""

        # Parse JSON from response (strip possible markdown fences)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        # Try to find JSON object in the text (DeepSeek with thinking may
        # output reasoning before the JSON)
        # Look for the first { and last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]

        result = json.loads(text)
        result["_usage"] = usage

        # Log token usage and cost to stderr for CI visibility
        reasoning_info = f", 推理={usage.get('reasoning_tokens', 0)}" if 'reasoning_tokens' in usage else ""
        cache_info = f", 缓存命中={cached_tokens}" if cached_tokens > 0 else ""
        cost_total = usage.get('cost_total', 0)
        import sys as _sys
        print(f"[DeepSeek] 输入={usage.get('prompt_tokens', '?')}, "
              f"输出={usage.get('completion_tokens', '?')}, "
              f"合计={usage.get('total_tokens', '?')}{reasoning_info}{cache_info} | "
              f"费用=¥{cost_total:.4f}",
              file=_sys.stderr)

        return result

    except json.JSONDecodeError as e:
        return {
            "summary": f"AI response parse error: {e}",
            "issues": [{
                "severity": "notice",
                "path": "$",
                "message": f"DeepSeek 返回的内容无法解析为 JSON: {e}",
                "suggestion": "重试或手动检查报告"
            }],
            "_raw_response": text[:500] if 'text' in dir() else "",
        }
    except Exception as e:
        error_name = type(e).__name__
        error_msg = str(e)
        # Detect common error patterns
        if "401" in error_msg or "Unauthorized" in error_msg or "auth" in error_msg.lower():
            return {
                "summary": f"DeepSeek API authentication failed: {error_msg}",
                "issues": [{
                    "severity": "notice",
                    "path": "$",
                    "message": f"DeepSeek API 认证失败: {error_msg}",
                    "suggestion": "检查 DEEPSEEK_API_KEY 是否有效"
                }]
            }
        elif "429" in error_msg or "rate" in error_msg.lower():
            return {
                "summary": f"DeepSeek API rate limited: {error_msg}",
                "issues": [{
                    "severity": "notice",
                    "path": "$",
                    "message": f"DeepSeek API 速率限制: {error_msg}",
                    "suggestion": "稍后重试或降低请求频率"
                }]
            }
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return {
                "summary": f"DeepSeek API timeout: {error_msg}",
                "issues": [{
                    "severity": "notice",
                    "path": "$",
                    "message": f"DeepSeek API 超时: {error_msg}",
                    "suggestion": "文件较大或服务繁忙，稍后重试"
                }]
            }
        else:
            return {
                "summary": f"AI analysis failed: {error_name}: {error_msg}",
                "issues": [{
                    "severity": "notice",
                    "path": "$",
                    "message": f"AI 语义分析异常: {error_name}: {error_msg}",
                    "suggestion": "检查 DeepSeek API 服务状态或稍后重试"
                }]
            }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def analyze_file(filepath: str, api_key: str) -> Dict[str, Any]:
    """Run AI semantic analysis on a single JSON file with full protection."""

    if not os.path.isfile(filepath):
        return {
            "file": filepath,
            "pass": True,
            "issues": [{"severity": "notice", "path": "$",
                        "message": f"File not found: {filepath}"}]
        }

    # --- Protection Layer 1: File size check ---
    size_error = check_file_size(filepath)
    if size_error:
        return {
            "file": filepath,
            "pass": False,
            "issues": [{"severity": "error", "path": "$",
                        "message": f"Token-bomb protection: {size_error}"}]
        }

    # --- Protection Layer 2: Parse JSON ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as e:
        return {
            "file": filepath,
            "pass": True,
            "issues": [{"severity": "error", "path": "$",
                        "message": f"Cannot read file: {e}"}]
        }

    # Check raw text size (before parsing, catch bombs that exploit streaming parsers)
    if len(raw_text) > MAX_FILE_SIZE_BYTES * 2:
        return {
            "file": filepath,
            "pass": False,
            "issues": [{"severity": "error", "path": "$",
                        "message": f"Token-bomb protection: raw file size "
                                   f"({len(raw_text) / 1_000_000:.1f}MB) exceeds limit"}]
        }

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {
            "file": filepath,
            "pass": True,
            "issues": [{"severity": "error", "path": "$",
                        "message": f"Cannot parse JSON: {e.msg} at line {e.lineno}"}]
        }

    # --- Protection Layer 3: Build safe focused view (scan + truncate) ---
    focused, security_issues = build_safe_focused_view(data, filepath)

    # --- Protection Layer 4: Token budget check ---
    user_prompt = (
        f"请分析以下验证报告 JSON 的语义质量问题：\n\n"
        f"文件: {filepath}\n\n"
        f"{json.dumps(focused, indent=2, ensure_ascii=False)}"
    )
    estimated_tokens = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(user_prompt)

    if estimated_tokens > MAX_ESTIMATED_TOKENS:
        return {
            "file": filepath,
            "pass": False,
            "issues": security_issues + [{
                "severity": "error",
                "path": "$",
                "message": f"Token-bomb protection: estimated prompt size "
                           f"({estimated_tokens} tokens) exceeds AI analysis limit "
                           f"({MAX_ESTIMATED_TOKENS})",
                "suggestion": "报告内容过大，请检查是否被注入了膨胀数据"
            }]
        }

    # Also check the total character count of the focused content
    focused_json_str = json.dumps(focused, ensure_ascii=False)
    if len(focused_json_str) > MAX_FOCUSED_CONTENT_CHARS:
        # Truncate the focused content aggressively
        return {
            "file": filepath,
            "pass": False,
            "issues": security_issues + [{
                "severity": "error",
                "path": "$",
                "message": f"Token-bomb protection: focused content size "
                           f"({len(focused_json_str) / 1_000:.0f}K chars) exceeds limit "
                           f"({MAX_FOCUSED_CONTENT_CHARS / 1_000:.0f}K)",
                "suggestion": "报告内容过大，请检查是否有异常的数据膨胀"
            }]
        }

    # --- Call DeepSeek API ---
    result = call_deepseek(SYSTEM_PROMPT, user_prompt, api_key)

    if result is None:
        result = {"summary": "AI analysis returned no result", "issues": []}

    # Merge security issues from protection layer with AI findings
    ai_issues = result.get("issues", [])
    all_issues = security_issues + ai_issues
    # Extract actual token usage from API response
    actual_usage = result.get("_usage", {})

    return {
        "file": filepath,
        "pass": True,  # AI analysis is advisory
        "ai_summary": result.get("summary", ""),
        "issues": all_issues,
        "_meta": {
            "estimated_tokens": estimated_tokens,
            "focused_chars": len(focused_json_str),
            "actual_usage": actual_usage,
        },
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps(
            {"error": "Usage: ai_quality_check.py <file1.json> [file2.json ...]"},
            indent=2, ensure_ascii=False))
        sys.exit(2)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print(json.dumps({
            "pass": True,
            "error": "DEEPSEEK_API_KEY not set. Set the environment variable to enable AI analysis.",
            "files": {
                f: {
                    "file": f,
                    "pass": True,
                    "issues": [{
                        "severity": "notice",
                        "path": "$",
                        "message": "AI 语义分析跳过：未设置 DEEPSEEK_API_KEY 环境变量",
                        "suggestion": "在 GitHub Secrets 中配置 DEEPSEEK_API_KEY 以启用 AI 分析"
                    }]
                }
                for f in sys.argv[1:]
            }
        }, indent=2, ensure_ascii=False))
        sys.exit(0)

    files = sys.argv[1:]
    results = {}

    for filepath in files:
        results[filepath] = analyze_file(filepath, api_key)

    output = {
        "pass": True,  # AI analysis is always advisory
        "files": results,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
