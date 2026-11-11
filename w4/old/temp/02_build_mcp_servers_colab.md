# Build an MCP server — Math and TripUtilities in Colab

This guide follows the steps in the [official **Build an MCP server** tutorial (2026-07-28)](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/docs/2026-07-28/develop/build-server.mdx). We're changing the example, not the purpose: you'll build two small MCP servers and then connect to them from a host.

The official tutorial uses a weather server and Claude for Desktop. **Our host is a simple Google Colab notebook**, and our servers are the two from your preparation activity:

- **Math**, with the tool `safe_calculator(expression: str)`.
- **TripUtilities**, with the tool `estimate_daily_budget(days: int, daily_cost: float)`.

Both use **stdio** and run as separate Python processes when a client connects. No LLM or API key is needed. If the host/client/server distinction is still fuzzy, start with [Architecture overview](01_mcp_architecture_colab.md).

> **Source note:** This is an adapted, easy-to-read teaching guide rather than a verbatim copy of the official page. The tool names and input arguments come from your uploaded notebook. Its `fastmcp.FastMCP` server wrapper is replaced with the official Python MCP SDK v2 `MCPServer`, so the server and Activity 1 client use the same SDK family.

## What we'll be building

We'll create **two files**: `math_server.py` and `trip_server.py`. The Colab host will launch them later through two MCP client connections.

## Core MCP concepts

MCP servers can provide **tools** (operations), **resources** (readable data), and **prompts** (reusable message templates). In this guide we're focusing on tools. Each tool is advertised with a name, description, and input schema. The client can discover that information before calling the tool.

## Prerequisite knowledge

Basic Python functions, type hints, and dictionaries are enough. Use a fresh Colab Python runtime. You'll run the code blocks in the order shown.

## Logging in MCP servers

A `stdio` MCP server reads protocol messages from **stdin** and sends protocol responses on **stdout**. Don't put ordinary `print()` diagnostics on the server's stdout: they can break the message stream. Use `logging` (which writes to stderr by default). This is different from printing output in the **Colab host**, which is fine.

## System requirements and environment setup

### Colab cell 1: prepare stderr before importing MCP

Run this **as the first cell in a fresh Colab runtime**, exactly as in your preparation notebook. It must appear before the first `mcp` import.

```python
import sys

# MCP launches its servers as subprocesses and hands them this notebook's
# stderr, which must have a real file descriptor. Notebook kernels do not
# provide one, so we point stderr at a log file. The MCP library captures
# stderr the moment it first loads, so this redirect must run before any
# MCP import, which is why it sits at the top of the setup cell.
sys.stderr = open("mcp_server.log", "w")
```

This is a Colab subprocess workaround, not an MCP protocol rule. It can move notebook error output into the log. If you need to see diagnostic output later, read `mcp_server.log` or restart the runtime to restore its usual streams.

### Colab cell 2: install the official Python SDK

```python
%pip -q install -U "mcp>=2,<3"
```

We don't need LangChain, the separate `fastmcp` package, `uvx`, Claude, or an API key for these local examples.

## Building your servers

### Math server — register the calculator tool

This is the supplied **Math** server in official SDK v2 form. `@mcp.tool()` registers the function for MCP discovery. `mcp.run(transport="stdio")` runs the server when it is launched as a separate program; it does **not** run merely because the notebook writes the file.

**Safety adjustment:** The attached calculator uses Python `eval`, which is not safe for untrusted expressions even with builtins restricted. This version only accepts numeric literals, basic arithmetic operators and parentheses, retaining the lesson's tool name and purpose.

---

```python
%%writefile math_server.py
import ast
import operator
from mcp.server import MCPServer

mcp = MCPServer("Math")

BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def evaluate_arithmetic(node):
    if isinstance(node, ast.Expression):
        return evaluate_arithmetic(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        if abs(node.value) > 1_000_000_000:
            raise ValueError("Number is too large")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](evaluate_arithmetic(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left = evaluate_arithmetic(node.left)
        right = evaluate_arithmetic(node.right)
        answer = BINARY[type(node.op)](left, right)
        if abs(answer) > 1_000_000_000:
            raise ValueError("Result is too large")
        return answer
    raise ValueError("Only numbers and +, -, *, /, parentheses are accepted")


@mcp.tool()
def safe_calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression with +, -, *, / and parentheses."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(evaluate_arithmetic(tree))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"Invalid arithmetic expression: {exc}") from exc


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

### TripUtilities server — register the budget tool

The TripUtilities implementation below follows the attached server almost line-for-line, with `MCPServer` substituted for the standalone framework. The tool accepts a **integer duration** and a **floating-point daily cost**, returning their product. Invalid negative values raise a Python exception inside the server; the SDK will communicate a tool failure when supported by the server implementation.

---

```python
%%writefile trip_server.py
from mcp.server import MCPServer

mcp = MCPServer("TripUtilities")


@mcp.tool()
def estimate_daily_budget(days: int, daily_cost: float) -> float:
    """Estimate a trip budget from duration and daily cost."""
    if days < 0 or daily_cost < 0:
        raise ValueError("Inputs must be nonnegative")
    return days * daily_cost


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## Understanding the server code

Identify the role of each part:

1. `MCPServer("Math")` constructs the server object and gives it a display identity.
2. `@mcp.tool()` announces a function as an MCP tool.
3. The function's name, annotations, and docstring support a **discoverable contract** (JSON Schema).
4. The function body performs the computation **only after a tool call**.
5. `mcp.run(transport="stdio")` listens for MCP protocol messages in the server process.

The host will **not** use `from math_server import safe_calculator`. It will send `tools/list` and `tools/call` to the process through the client SDK.

**Logging rule:** Server-side `print()` to standard output is unsafe over stdio because stdout is the protocol channel. Use standard Python `logging` to stderr for operational messages. The code above deliberately has no debug `print()` in its server paths.

**Think:** Would merely writing the two `.py` files create running servers? **No.** The client launches each when entering its stdio connection context.

---

## Testing your servers

The two `%%writefile` cells only **create the files**. They do not launch either MCP server. Let's check that the scripts exist and are valid Python before connecting.

```python
from pathlib import Path

for filename in ("math_server.py", "trip_server.py"):
    script = Path(filename)
    assert script.exists(), f"Missing {filename}: run its %%writefile cell"
    compile(script.read_text(encoding="utf-8"), filename, "exec")
    print(f"Created {filename} and checked Python syntax")
```

You don't need to run the server programs in separate notebook cells. In the next guide, `async with Client(MATH_SERVER)` or `async with Client(TRIP_SERVER)` launches the corresponding subprocess for you.

### Common troubleshooting

If the server cannot start, check that the file exists, that the SDK is installed in the active runtime, and that the server diagnostic log contains useful details. If a tool raises a `ValueError`, that's a **tool execution** issue, not necessarily a failure to launch the server.

**Next:** [Build an MCP client](03_build_mcp_client_colab.md) shows how to discover these two tools and execute them. Keep the same Colab runtime open, because the client needs the `math_server.py` and `trip_server.py` files you just wrote.
