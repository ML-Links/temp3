# Connect Two MCP Servers with Python, LangChain, and an LLM

*A three-stage, Python-only implementation based on the attached `w4_nn_project_prep2(1).ipynb` and **Activity 1: MCP Clients over HTTP and stdio**.*

**Updated:** 10 November 2026  
**Python:** 3.11 or newer recommended  
**Transport:** local MCP servers over **stdio**

## What this project builds

The notebook provides two MCP servers. We will use **both** in every stage:

| Server | Python file | MCP tool | Purpose |
|---|---|---|---|
| Math | `math_server.py` | `safe_calculator(expression: str)` | Calculate arithmetic expressions. |
| TripUtilities | `trip_server.py` | `estimate_daily_budget(days: int, daily_cost: float)` | Calculate a simple trip budget. |

The three stages are:

1. **Python + MCP, no LangChain, no LLM:** create clients, connect to both servers, discover tools, and call tools explicitly.
2. **Python + LangChain, no LLM:** convert discovered MCP tools into LangChain tools and call them explicitly.
3. **Python + LangChain + LLM:** give both tools to a LangChain agent; the model selects tools in response to a user question.

```text
Stage 1: Your Python code  -> FastMCP clients -> Math and Trip MCP servers
Stage 2: Your Python code  -> LangChain MCPAdapter -> Math and Trip MCP servers
Stage 3: User question -> LangChain agent + LLM
                                |        |
                                +--> Math MCP tool
                                +--> Trip MCP tool
                                |
                          Final answer to user
```

### Why the LangChain code differs from the notebook

The notebook imports `MultiServerMCPClient` from `langchain_mcp_adapters`. That **older package is archived** and its latest published version requires MCP Python SDK **1.x**; it is not compatible in one environment with newer FastMCP configurations using MCP SDK **2.x**. Current LangChain instead provides **`MCPAdapter`**, through `langchain[mcp]` (available in LangChain 1.4+). This guide uses the newer approach for an internally consistent install. The core idea from your notebook—**one client/adapter providing tools from two servers**—remains the same.

This is also why you should **not** copy the Activity 1 installation requirement `mcp>=2,<3` into an environment that installs the archived `langchain-mcp-adapters` package.

## 0. Folder layout

Save each following Python block into a file of the specified name, all in **one directory**:

```text
mcp_project/
    install_packages.py
    math_server.py
    trip_server.py
    mcp_config.py
    01_python_mcp_client.py
    02_langchain_without_llm.py
    03_langchain_with_llm.py
```

There are **no notebook magics** (`%pip`, `%%writefile`), JavaScript files, or separate server-start commands. The clients launch each server automatically as a subprocess. Run the three numbered scripts using Python, in order.

## 1. Install dependencies — `install_packages.py`

```python
import subprocess
import sys
from importlib.metadata import version

# Keep this project on one compatible generation of MCP/LangChain packages.
packages = [
    "fastmcp>=4.1,<5",
    "langchain[mcp]>=1.4,<2",
    "langchain-openai>=1.7,<2",
]

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
)

for package in ("fastmcp", "langchain", "langchain-openai"):
    print(f"{package}: {version(package)}")
```

Run this Python file once, then use the same Python environment/interpreter for every subsequent file. Stages 1 and 2 do not require an LLM API key; only Stage 3 does. If packages were already imported in a notebook kernel, restart the kernel after installation.

## 2. MCP server A — `math_server.py`

This preserves the notebook's `safe_calculator` tool and supported operations, but **does not use `eval`**. Restricting the syntax is important because an LLM or user can control the `expression` argument.

```python
import ast
import math
import operator

from fastmcp import FastMCP

mcp = FastMCP("Math")

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate_node(node):
    if isinstance(node, ast.Expression):
        return calculate_node(node.body)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        left = calculate_node(node.left)
        right = calculate_node(node.right)
        return BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](calculate_node(node.operand))

    raise ValueError("Only numbers, parentheses, +, -, *, and / are allowed")


@mcp.tool()
def safe_calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression, e.g. '8273 * 1821'.

    Supports numeric literals, parentheses, +, -, *, and / only.
    """
    if not expression.strip() or len(expression) > 200:
        raise ValueError("Expression must be between 1 and 200 characters")

    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("Expression is too complex")

    value = calculate_node(tree)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result must be finite")
    return str(value)


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

For example, `8273 * 1821` should evaluate to `15065133`. Unsupported expressions or division by zero become MCP tool errors rather than arbitrary Python code execution.

## 3. MCP server B — `trip_server.py`

This retains the TripUtilities server and the budget calculation from the notebook.

```python
from fastmcp import FastMCP

mcp = FastMCP("TripUtilities")


@mcp.tool()
def estimate_daily_budget(days: int, daily_cost: float) -> float:
    """Estimate a trip budget from a number of days and cost per day."""
    if days < 0 or daily_cost < 0:
        raise ValueError("Inputs must be nonnegative")
    return days * daily_cost


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Example: 5 days at 120 per day returns `600.0`. The numbers have no currency unit inside the function; use the same currency for input and output.

## 4. Shared server configuration — `mcp_config.py`

The same paths and subprocess commands are reused in all three stages. Using `sys.executable` and absolute script paths avoids accidentally launching a different Python interpreter or a script from the wrong working directory.

```python
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent


def server_specs():
    files = {
        "math": PROJECT_DIR / "math_server.py",
        "trip": PROJECT_DIR / "trip_server.py",
    }

    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} server: {path}")

    return {
        name: {
            "command": sys.executable,
            "args": [str(path)],
        }
        for name, path in files.items()
    }


def langchain_mcp_config():
    # Current LangChain's MCPAdapter accepts this FastMCP/MCPConfig shape.
    return {"mcpServers": server_specs()}
```

---

## Stage 1 — Pure Python MCP clients (no LangChain, no LLM)

**File:** `01_python_mcp_client.py`

This follows the Activity 1 sequence: **open connection → discover tool schemas → invoke tool → inspect actual result**. `AsyncExitStack` keeps both clients open at once. The two servers communicate over stdio, so they do **not** need separate HTTP ports or manual startup.

```python
import asyncio
import json
from contextlib import AsyncExitStack

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from mcp_config import server_specs


async def main():
    specs = server_specs()

    async with AsyncExitStack() as stack:
        clients = {}

        # Connect to both independent server subprocesses.
        for name, spec in specs.items():
            transport = StdioTransport(
                command=spec["command"],
                args=spec["args"],
            )
            clients[name] = await stack.enter_async_context(Client(transport))
            print(f"Connected to {name} MCP server")

        # Discovery only: listing tools does not execute them.
        for name, client in clients.items():
            tools = await client.list_tools()
            print(f"\n{name} server tools:")
            for tool in tools:
                print("Tool:", tool.name)
                print("Description:", tool.description)
                schema = tool.model_dump(by_alias=True).get("inputSchema", {})
                print("Input schema:", json.dumps(schema, indent=2))

        # Explicitly choose a tool and supply its schema-compatible arguments.
        math_result = await clients["math"].call_tool(
            "safe_calculator",
            {"expression": "8273 * 1821"},
            raise_on_error=False,
        )
        trip_result = await clients["trip"].call_tool(
            "estimate_daily_budget",
            {"days": 5, "daily_cost": 120.0},
            raise_on_error=False,
        )

        print("\nMath error flag:", math_result.is_error)
        print("Math result:", math_result.data if not math_result.is_error
              else math_result.content)

        print("Trip error flag:", trip_result.is_error)
        print("Trip result:", trip_result.data if not trip_result.is_error
              else trip_result.content)


if __name__ == "__main__":
    asyncio.run(main())
```

**Expected result when both servers are working:** the Math result represents `15065133` and the Trip result represents `600.0`. Exact result formatting can vary by SDK version. If `is_error` is `True`, do not treat that tool output as a successful calculation.

**What happens:** Your Python code decides which tools to call. MCP discovers schemas and carries requests/results; no LLM is involved.

---

## Stage 2 — Python + LangChain MCP tools (still no LLM)

**File:** `02_langchain_without_llm.py`

Now `MCPAdapter` discovers both servers' tools and converts them into **LangChain tools**. We still select and invoke the tools explicitly in Python; using LangChain does **not** automatically imply an LLM.

```python
import asyncio

from langchain.mcp import MCPAdapter
from mcp_config import langchain_mcp_config


async def main():
    async with MCPAdapter(langchain_mcp_config()) as adapter:
        tools = await adapter.list_tools()

        print("LangChain tools discovered from both servers:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            print("  Arguments:", tool.args)

        # In a multi-server MCPConfig, names are prefixed by server key.
        by_name = {tool.name: tool for tool in tools}

        math_result = await by_name["math_safe_calculator"].ainvoke(
            {"expression": "(30 + 12) * 2"}
        )
        trip_result = await by_name["trip_estimate_daily_budget"].ainvoke(
            {"days": 5, "daily_cost": 120.0}
        )

        print("\nLangChain Math result:", math_result)
        print("LangChain Trip result:", trip_result)


if __name__ == "__main__":
    asyncio.run(main())
```

**Expected calculation:** `(30 + 12) * 2` equals `84` and the trip budget equals `600.0`. The result objects/text may be formatted differently by LangChain; inspect the returned content.

**What changed from Stage 1?** Instead of calling a FastMCP client's `call_tool()`, we call a LangChain tool's `ainvoke()`. The server implementations are untouched. Still **no OpenAI key** and **no model**.

---

## Stage 3 — Python + LangChain + an LLM agent

**File:** `03_langchain_with_llm.py`

This stage adds an OpenAI chat model through `langchain-openai`, and uses LangChain's `create_agent()` so the LLM can choose the correct tool based on the request. **Only this stage needs** an `OPENAI_API_KEY` with access/billing enabled for the selected model.

```python
import asyncio
import getpass
import os

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from mcp_config import langchain_mcp_config


def ensure_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass(
            "Enter your OpenAI API key (hidden): "
        )
    if not os.environ["OPENAI_API_KEY"]:
        raise ValueError("OPENAI_API_KEY is required for Stage 3")


async def main():
    ensure_openai_key()

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )

    async with MCPAdapter(langchain_mcp_config()) as adapter:
        tools = await adapter.list_tools()
        print("Tools offered to the LLM:", [tool.name for tool in tools])

        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=(
                "You are a helpful trip-budget assistant. "
                "Use trip_estimate_daily_budget for daily-cost trip budgets. "
                "Use math_safe_calculator for arithmetic, including contingencies. "
                "Base numeric conclusions on tool results, not guesses. "
                "Use the user's currency unit; do not invent exchange rates."
            ),
        )

        question = (
            "I will travel for 5 days and spend 120 euros per day. "
            "First use the trip budget tool to calculate my base budget. "
            "Then use the calculator tool to add a 12% contingency. "
            "What is the base budget, contingency amount, and total in euros?"
        )

        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        # Show the real tool-call trail rather than assuming the model used them.
        print("\n--- Actual tool activity ---")
        for message in response["messages"]:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    print("LLM requested:", tool_call["name"], tool_call["args"])
            elif isinstance(message, ToolMessage):
                print("MCP tool replied:", message.name, message.content)

        print("\n--- Final model answer ---")
        print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
```

**Expected arithmetic if the agent calls the tools correctly:** base budget = **€600**, contingency = **€72**, total = **€672**. LLM wording and exact tool-call sequence can vary. The printed trail is the evidence of which tools were actually invoked; do not claim successful tool use if that trail or tool results show otherwise.

**What changed from Stage 2?** The application no longer selects every tool itself. The LLM receives tool descriptions and JSON schemas, generates tool calls, and then reads tool results through LangChain. The LLM does **not** directly import or execute the server functions.

---

## 5. How to run and verify

1. Put all seven `.py` files in the same directory (for example, `mcp_project`).
2. Run **`install_packages.py`** in that Python environment, then run **`01_python_mcp_client.py`**. Verify both tool catalogs and both returned values.
3. Run **`02_langchain_without_llm.py`**. Verify that both prefixed LangChain tools are listed and callable.
4. Run **`03_langchain_with_llm.py`**. Provide your API key securely when prompted (or set `OPENAI_API_KEY` in the process environment beforehand). Check both the tool-call trail and the final answer.

For example, launch a file through your IDE's **Run** action or use Python itself to execute `python 01_python_mcp_client.py`, and similarly for the others. You do **not** start `math_server.py` or `trip_server.py` separately when using stdio: the client manages each subprocess.

If you are using **Google Colab**, the scripts can still be used, but their relative locations must be adjusted to your notebook working directory. When calling an `async def main()` directly **inside** a running notebook event loop, use `await main()` instead of `asyncio.run(main())`. Activity 1's `mcp_server.log` redirection is a **Colab-specific subprocess diagnostic workaround**, not an MCP protocol requirement; use it only if notebook stderr causes subprocess-launch errors, and perform the redirection before importing MCP modules.

## 6. Common problems

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: langchain.mcp` | You installed an older LangChain; install `langchain[mcp]>=1.4` and restart the Python process. |
| Dependency resolution error for `mcp` | Do not mix `langchain-mcp-adapters` 0.3.x / MCP SDK 1.x with a modern MCP SDK 2.x / FastMCP 4.x installation. |
| No tools discovered / server stops immediately | Confirm both server paths exist, use the same interpreter, and inspect server stderr/logs. |
| `KeyError: math_safe_calculator` | Print actual discovered tool names; check that the multi-server `mcpServers` config was used. |
| `ToolError` or tool result has an error | Review the tool's arguments and exception; a tool-execution error differs from a failed connection. |
| Python prints unwanted text from a server | For stdio servers, do not print debug messages to stdout; write diagnostics to stderr/logging. |
| OpenAI authentication / model error | Check the API key, API billing/access, model name, and network connection. Stage 1 and 2 remain usable without it. |
| The model does not call both tools | Inspect the printed call trail, tighten the task/system instructions, and retry; model tool choice is not guaranteed by descriptions alone. |

## 7. What was adopted from your files?

- **From the notebook:** `FastMCP` servers named **Math** and **TripUtilities**, with tools `safe_calculator` and `estimate_daily_budget`, plus the idea of merging both servers' tools for LangChain.
- **From Activity 1:** the distinction between an **MCP host/client/server**, **stdio subprocesses**, **tool discovery vs. invocation**, JSON input schemas, `async with` / `await`, actual result inspection, and separation of **tool errors** from **connection errors**.
- **Updates introduced here:** a safer arithmetic parser instead of `eval`, one reusable server configuration, current `langchain.mcp.MCPAdapter` rather than the archived adapter, and a real LLM-driven third stage.

The Math and Trip examples are **local stdio servers**. Activity 1 also demonstrates remote **Streamable HTTP** servers (and local Time/Fetch providers); those are useful comparisons but are **not** the two servers defined in your notebook.

## Official reference links

- [LangChain: MCP integration](https://docs.langchain.com/oss/python/langchain/mcp)
- [LangChain: MCP connections and multiple servers](https://docs.langchain.com/oss/python/langchain/mcp/connections)
- [LangChain: Agents / `create_agent`](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain: OpenAI model integration](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [FastMCP: Client transports](https://gofastmcp.com/clients/transports)
- [FastMCP: Calling tools](https://gofastmcp.com/clients/tools)
- [MCP SDK: Client transports](https://py.sdk.modelcontextprotocol.io/client/transports/)

> **Execution note:** The examples are designed against current published APIs and their Python syntax has been checked, but no live MCP subprocesses or OpenAI API calls were executed in preparing this document. Test in your Python environment and retain the actual outputs/errors as Activity 1 recommends.
