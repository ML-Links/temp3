Yes. Your fine-tuning pipeline can stay essentially **unchanged**; the main change is the UI layer. Your current lab treats fine-tuning, JSON output, and Gradio as one connected workflow, so I would replace the Gradio-specific portions with a Streamlit application while keeping the Qwen/LoRA inference logic intact. 

Your existing Streamlit/Colab setup is also a good fit because it launches `app.py` on port `8501` and embeds that server directly into the Colab notebook.

## What I would change

There are four main changes:

1. **Install Streamlit** alongside the ML dependencies.
2. Keep the model training and inference cells unchanged.
3. Replace **Cells 10–12** with Streamlit versions.
4. Add your working **Streamlit Colab launcher** so the UI appears inside the notebook.

The original Cell 10 currently uses `gr.Interface(...)` and `demo.launch(...)`.  Cells 11 and 12 do the same thing for the Python-enforced JSON and Pydantic approaches. 

---

# Revised Section 3: Environment Setup & Application Launch

I would put this **near the beginning of the lab**, after the normal ML dependencies are installed.

## 3. Environment Setup & Application Launch

This lab uses **Streamlit** instead of Gradio for the interactive application.

The model training workflow remains the same: Qwen 2.5 is fine-tuned with LoRA, the resulting adapter is loaded for inference, and the application sends user prompts through the fine-tuned model.

Streamlit provides the user interface and displays the model's structured JSON response.

### Colab Streamlit Setup

Execute the following cell in your Google Colab notebook to install Streamlit, create the application, launch the background server, and render the interface inside Colab.

```python
# [Cell 0b] Streamlit Environment Setup

# 1. Install Streamlit and visualization libraries
!pip install -q streamlit plotly seaborn matplotlib

# 2. Write a baseline Streamlit application
with open("app.py", "w") as f:
    f.write("""import streamlit as st

st.set_page_config(
    page_title="MediCore Fine-Tuned Qwen",
    layout="centered"
)

st.title("MediCore Fine-Tuned Qwen Bot")

st.success("Streamlit server initialized successfully.")

st.write(
    "The application is ready. Continue through the lab "
    "to connect the fine-tuned model."
)
""")

# 3. Terminate any previous Streamlit instances cleanly
import subprocess
import time

subprocess.run(
    ["pkill", "-f", "streamlit"],
    stderr=subprocess.DEVNULL
)

# 4. Launch Streamlit in the background
subprocess.Popen(
    [
        "streamlit",
        "run",
        "app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# 5. Give the server time to start
time.sleep(3)

# 6. Generate a Colab proxy URL
from google.colab import output
from google.colab.output import eval_js

proxy_url = eval_js(
    "google.colab.kernel.proxyPort(8501)"
)

print("=" * 70)
print("FULL-SCREEN DIRECT URL (OPTIONAL):")
print(proxy_url)
print("=" * 70)

# 7. Embed Streamlit directly in the notebook
output.serve_kernel_port_as_iframe(
    8501,
    height="650"
)
```

### What this cell does

The cell performs five tasks:

1. Installs Streamlit.
2. Creates an `app.py` file.
3. Stops any previous Streamlit process.
4. Starts Streamlit on port `8501`.
5. Embeds the Streamlit application inside the Colab notebook.

You can also open the printed **FULL-SCREEN DIRECT URL** in a separate browser tab.

> **Important:** The Streamlit server should be started only once. Later cells in the lab will update `app.py` and restart the server when necessary.

---

# Replace the Gradio section

I would also rename the existing section:

**Current:**

> `## Core project extension: structured JSON and Gradio`

**New:**

> `## Core project extension: structured JSON and Streamlit`

The concepts remain the same. The original lab correctly emphasizes that JSON is useful when the model output needs to be consumed programmatically. 

---

# Replace Cell 10 with Streamlit

The original Cell 10 uses a system prompt to ask Qwen to produce:

```json
{"answer": "..."}
```

That inference logic should **not change**. Only the UI changes.

Here's the Streamlit equivalent.

### Strategy 1: Model-Controlled JSON with Streamlit

This version relies on the model following a system prompt that requests valid JSON.

```python
# [Cell 10] Streamlit Interface — Model-Controlled JSON

import json

# ---------------------------------------------------------
# Model inference function
# ---------------------------------------------------------

def generate_response(user_prompt):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "You must ONLY answer in valid JSON format "
                'using the following structure: '
                '{"answer": "your detailed response here"}'
            )
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    # Apply Qwen ChatML template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Convert prompt to tensors and move to GPU
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    # Remove the original prompt
    generated_ids = output[0][
        inputs.input_ids.shape[1]:
    ]

    # Convert generated tokens back to text
    response_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return response_text


# ---------------------------------------------------------
# Create Streamlit application
# ---------------------------------------------------------

with open("app.py", "w") as f:

    f.write("""
import streamlit as st
import json

st.set_page_config(
    page_title="MediCore Fine-Tuned Qwen",
    layout="centered"
)

st.title("MediCore Fine-Tuned Qwen Bot")

st.write(
    "Ask questions about MediCore Hospital. "
    "The model is instructed to reply in JSON format."
)

user_prompt = st.text_area(
    "Enter your prompt here",
    value="Who leads the neurology department at MediCore Hospital?",
    height=120
)

if st.button("Generate Response"):

    if not user_prompt.strip():
        st.warning("Please enter a prompt.")

    else:

        with st.spinner("Generating response..."):

            # The model inference code is executed
            # in the Streamlit process.
            result = generate_response(user_prompt)

        st.subheader("Model Output")

        st.code(
            result,
            language="json"
        )
""")
```

### Important note

Because Streamlit executes `app.py`, the variables `model` and `tokenizer` need to be available **inside the Streamlit process**.

Therefore, the recommended deployment pattern is to put the model-loading code directly into `app.py`, rather than relying on variables that exist only in the Colab notebook.

There is one important correction to make here: **the simple `generate_response()` function cannot live only in the Colab notebook while `app.py` runs as a separate process**. The Streamlit process needs its own access to the model.

So for the lab, I recommend making the Streamlit application **self-contained**.

---

# Recommended Streamlit architecture

Instead of trying to pass the notebook's `model` object into Streamlit, have `app.py` load the saved LoRA model.

That gives you this architecture:

```text
Google Colab
│
├── Notebook
│   ├── Load base model
│   ├── Fine-tune with LoRA
│   ├── Save ./my_qwen
│   └── Test model
│
└── Streamlit app.py
    │
    ├── Load Qwen base model
    ├── Load ./my_qwen LoRA adapter
    ├── Receive user prompt
    ├── Generate response
    └── Display JSON
```

This fits the existing lab's LoRA workflow because `trainer.save_model("./my_qwen")` saves the adapter, and the later loading step attaches that adapter to the original Qwen model. 

---

# Better Cell 10

I would therefore use this as the **actual student-facing Cell 10**.

### Strategy 1: Model-Controlled JSON with Streamlit

```python
# [Cell 10] Build the Streamlit Application

with open("app.py", "w") as f:

    f.write(r'''
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="MediCore Fine-Tuned Qwen",
    page_icon="🏥",
    layout="centered"
)


# =========================================================
# Load model
# =========================================================

@st.cache_resource
def load_model():

    model_path = "./my_qwen"

    # Read LoRA configuration
    from peft import PeftConfig

    config = PeftConfig.from_pretrained(model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path
    )

    tokenizer.pad_token = tokenizer.eos_token

    # Load original Qwen model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="cuda",
        torch_dtype=torch.float16
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        model_path
    )

    model.config.use_cache = True

    return model, tokenizer


model, tokenizer = load_model()


# =========================================================
# Model inference
# =========================================================

def generate_response(user_prompt):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "You must ONLY answer in valid JSON format "
                'using the following structure: '
                '{"answer": "your detailed response here"}'
            )
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = output[0][
        inputs.input_ids.shape[1]:
    ]

    response_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    return response_text


# =========================================================
# Streamlit UI
# =========================================================

st.title("🏥 MediCore Fine-Tuned Qwen Bot")

st.write(
    "Ask questions about MediCore Hospital. "
    "The fine-tuned Qwen model is instructed to "
    "return its answer in JSON format."
)

user_prompt = st.text_area(
    "Enter your prompt",
    value="Who leads the neurology department at MediCore Hospital?",
    height=120
)

if st.button("Generate Response", type="primary"):

    if not user_prompt.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating response..."):

            response = generate_response(user_prompt)

        st.subheader("Model Output")

        # Try to parse the model response as JSON
        try:

            parsed = json.loads(response)

            st.json(parsed)

        except Exception:

            # If the model failed to produce valid JSON,
            # show the raw output instead.
            st.warning(
                "The model did not return valid JSON."
            )

            st.code(
                response,
                language="text"
            )
''')
```

> **Important:** Add `import json` near the top of `app.py` if you use the JSON parsing section above.

Actually, to avoid that small omission, I would use the following import line in the final version:

```python
import streamlit as st
import json
import torch
```

---

# Cell 11 — Python-enforced JSON

Your lab says this is the recommended default because Python guarantees the wrapper structure. 

This maps particularly cleanly to Streamlit.

### Strategy 2: Python-Enforced JSON with Streamlit

Instead of asking the model to generate JSON, the model generates normal text and Python wraps that response inside a JSON object.

```python
# [Cell 11] Streamlit — Python-Enforced JSON

with open("app.py", "w") as f:

    f.write(r'''
import streamlit as st
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    PeftModel,
    PeftConfig
)


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="MediCore Fine-Tuned Qwen",
    page_icon="🏥",
    layout="centered"
)


# =========================================================
# Load model
# =========================================================

@st.cache_resource
def load_model():

    path = "./my_qwen"

    config = PeftConfig.from_pretrained(path)

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path
    )

    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="cuda",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(
        base_model,
        path
    )

    model.config.use_cache = True

    return model, tokenizer


model, tokenizer = load_model()


# =========================================================
# Generate response
# =========================================================

def generate_response(user_prompt):

    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = output[0][
        inputs.input_ids.shape[1]:
    ]

    response_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    # Python-enforced JSON structure
    return {
        "answer": response_text
    }


# =========================================================
# Streamlit UI
# =========================================================

st.title("🏥 MediCore Fine-Tuned Qwen Bot")

st.write(
    "Ask questions about MediCore Hospital."
)

user_prompt = st.text_area(
    "Enter your prompt",
    value="Who leads the neurology department at MediCore Hospital?",
    height=120
)

if st.button("Generate Response", type="primary"):

    if not user_prompt.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating response..."):

            result = generate_response(user_prompt)

        st.subheader("JSON Output")

        st.json(result)
''')
```

This is probably the version I would make the **main recommended Streamlit path** in your lab.

---

# Cell 12 — Pydantic + Streamlit

The existing Pydantic approach creates a `HospitalResponse` object and serializes it to JSON. 

The Streamlit equivalent is:

### Strategy 3: Pydantic Structured Output with Streamlit

```python
# [Cell 12] Streamlit — Pydantic Structured Output

with open("app.py", "w") as f:

    f.write(r'''
import streamlit as st
import torch

from pydantic import BaseModel, Field

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import (
    PeftModel,
    PeftConfig
)


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="MediCore Fine-Tuned Qwen",
    page_icon="🏥",
    layout="centered"
)


# =========================================================
# Pydantic schema
# =========================================================

class HospitalResponse(BaseModel):

    answer: str = Field(
        description="The main text answer to the user's question"
    )

    model_version: str = Field(
        default="Qwen2.5-1.5B-MediCore",
        description="The model used"
    )


# =========================================================
# Load model
# =========================================================

@st.cache_resource
def load_model():

    path = "./my_qwen"

    config = PeftConfig.from_pretrained(path)

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path
    )

    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="cuda",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(
        base_model,
        path
    )

    model.config.use_cache = True

    return model, tokenizer


model, tokenizer = load_model()


# =========================================================
# Generate response
# =========================================================

def generate_response(user_prompt):

    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = output[0][
        inputs.input_ids.shape[1]:
    ]

    raw_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    structured_response = HospitalResponse(
        answer=raw_text
    )

    return structured_response.model_dump()


# =========================================================
# Streamlit UI
# =========================================================

st.title("🏥 MediCore Fine-Tuned Qwen Bot")

st.write(
    "Ask questions about MediCore Hospital. "
    "The response is validated using a Pydantic schema."
)

user_prompt = st.text_area(
    "Enter your prompt",
    value="Who leads the neurology department at MediCore Hospital?",
    height=120
)

if st.button("Generate Response", type="primary"):

    if not user_prompt.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating response..."):

            result = generate_response(user_prompt)

        st.subheader("Structured JSON Output")

        st.json(result)
''')
```

---

# Add a final Streamlit launch cell

After writing the new `app.py`, you need to restart Streamlit.

I recommend making this a separate cell rather than repeating the launch code in every UI cell.

### Launch the Streamlit Application

Run this cell after creating or updating `app.py`.

```python
# [Cell 13] Launch Streamlit

import subprocess
import time

# Stop existing Streamlit processes
subprocess.run(
    ["pkill", "-f", "streamlit"],
    stderr=subprocess.DEVNULL
)

time.sleep(1)

# Start Streamlit
subprocess.Popen(
    [
        "streamlit",
        "run",
        "app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

time.sleep(3)

# Create Colab proxy URL
from google.colab import output
from google.colab.output import eval_js

proxy_url = eval_js(
    "google.colab.kernel.proxyPort(8501)"
)

print("=" * 70)
print("FULL-SCREEN STREAMLIT URL:")
print(proxy_url)
print("=" * 70)

# Display Streamlit inside Colab
output.serve_kernel_port_as_iframe(
    8501,
    height="700"
)
```

---

## One important change to the lab's workflow

I would change the workflow summary from:

> `connect the result to Gradio`

to:

> `connect the result to Streamlit`

and the overall workflow becomes:

```text
1. Prepare Colab GPU
       ↓
2. Install ML dependencies + Streamlit
       ↓
3. Load Qwen + tokenizer
       ↓
4. Baseline inference
       ↓
5. Load and preprocess MediCore dataset
       ↓
6. Attach LoRA adapters
       ↓
7. Train adapters
       ↓
8. Test fine-tuned model
       ↓
9. Produce structured JSON
       ↓
10. Build Streamlit application
       ↓
11. Launch Streamlit inside Colab
       ↓
12. Optional: merge LoRA adapter
```

The underlying fine-tuning steps do **not** need to change. Your existing Step 7 already performs the correct ChatML-based inference and greedy decoding. 

### Also change these references throughout the Markdown

| Current wording       | Replace with             |
| --------------------- | ------------------------ |
| `Gradio`              | `Streamlit`              |
| `Gradio interface`    | `Streamlit interface`    |
| `Gradio UI`           | `Streamlit UI`           |
| `gr.Interface`        | Streamlit UI components  |
| `gr.Textbox`          | `st.text_area`           |
| `gr.Code`             | `st.code` / `st.json`    |
| `demo.launch()`       | Streamlit server launch  |
| `share=True`          | Colab proxy URL          |
| `rendering in Gradio` | `rendering in Streamlit` |

The original lab's **JSON strategies themselves should remain**: model-controlled JSON, Python-enforced JSON, and Pydantic structured output. Only their presentation layer changes. 

**My recommendation:** make **Python-enforced JSON + Streamlit (Cell 11)** the primary student path. It is the simplest and most reliable combination: the model concentrates on answering the question, while Python guarantees that Streamlit receives a valid JSON structure. The lab itself already identifies this approach as the safest default. 

If you want, I can also **rewrite the entire uploaded `activity1(1).md` into a clean Streamlit-only version**, preserving the existing explanations, cell numbering, Q&A sections, and appendices while removing all Gradio references.
