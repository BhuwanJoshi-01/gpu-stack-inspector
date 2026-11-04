// GPU Stack Inspector project page: the case switcher of figure 1 and the copy button.
// SPDX-License-Identifier: MIT
//
// The first case is real output from the test laptop (docs/sample-reports/compare-cpu-only.md);
// the others are what the same rules say for other inputs. Finding text is the tool's own.

"use strict";

const GPU = ["GeForce RTX 3050 Laptop GPU", "compute capability 8.6"];
const DRIVER = ["595.99.02", "supports CUDA up to 13.2"];
const UNSET = ["CUDA_VISIBLE_DEVICES unset", "the GPUs are visible to programs started here"];
const CONDA_A = {
    path: "~/envs/sapiens2",
    python: ["Python 3.13.13", "conda env"],
    torch: ["torch 2.11.0+cu128", "CUDA 12.8 build"],
    libs: ["15 nvidia-* wheels", "cuBLAS, cuDNN, NCCL and more"],
};

const CASES = {
    cpu: {
        a: CONDA_A,
        b: {
            path: "~/envs/cpu-venv",
            python: ["Python 3.12.13", "uv venv"],
            torch: ["torch 2.15.0+cpu", "CPU-only build", "fail", "GSI-TORCH-002"],
            libs: ["none", "no CUDA libraries came with this build", "empty"],
        },
        shell: UNSET,
        driver: DRIVER,
        gpu: GPU,
        command: "gpu-stack-inspector compare \\\n  --python ~/envs/sapiens2/bin/python \\\n  --python ~/envs/cpu-venv/bin/python",
        findings: [
            ["fail", "GSI-TORCH-002", "B", "torch 2.15.0+cpu is a CPU-only build",
                "This torch has no CUDA support at all: torch.cuda.is_available() is False whatever the driver can do.",
                "Install a CUDA build from the selector at pytorch.org, for CUDA 13.2 or older."],
            ["warn", "GSI-DIFF-001", "comparison", "The environments have different PyTorch builds: A CUDA 12.8, B CPU only",
                "This is the most likely reason CUDA works in one environment and not in another.",
                "Install the same CUDA build of torch in the environment that should use the GPU."],
        ],
    },
    driver: {
        a: {
            path: "~/envs/train",
            python: ["Python 3.12.11", "venv"],
            torch: ["torch 2.8.0+cu128", "CUDA 12.8 build"],
            libs: ["nvidia-* wheels", "for CUDA 12"],
        },
        b: {
            path: "~/envs/nightly",
            python: ["Python 3.13.7", "uv venv"],
            torch: ["torch 2.10.0+cu130", "CUDA 13.0 build", "fail", "GSI-CUDA-003"],
            libs: ["nvidia-* wheels", "for CUDA 13"],
        },
        shell: UNSET,
        driver: ["570.172.08", "supports CUDA up to 12.8"],
        gpu: GPU,
        command: "gpu-stack-inspector compare \\\n  --python ~/envs/train/bin/python \\\n  --python ~/envs/nightly/bin/python",
        findings: [
            ["fail", "GSI-CUDA-003", "B", "torch is built for CUDA 13.0, newer than the driver supports",
                "The driver supports CUDA up to 12.8. CUDA 13.x needs driver 580 or newer. CUDA will not initialise.",
                "Update the NVIDIA driver, or install a torch build for CUDA 12.8 or older."],
            ["warn", "GSI-DIFF-001", "comparison", "The environments have different PyTorch builds: A CUDA 12.8, B CUDA 13.0",
                "This is the most likely reason CUDA works in one environment and not in another.",
                "Install the same CUDA build of torch in the environment that should use the GPU."],
        ],
    },
    hidden: {
        a: CONDA_A,
        b: {
            path: "~/envs/finetune",
            python: ["Python 3.12.13", "uv venv"],
            torch: ["torch 2.11.0+cu128", "CUDA 12.8 build"],
            libs: ["15 nvidia-* wheels", "cuBLAS, cuDNN, NCCL and more"],
        },
        shell: ['CUDA_VISIBLE_DEVICES=""', "set to an empty value", "warn", "GSI-ENV-003"],
        driver: DRIVER,
        gpu: GPU,
        command: "gpu-stack-inspector compare \\\n  --python ~/envs/sapiens2/bin/python \\\n  --python ~/envs/finetune/bin/python",
        findings: [
            ["warn", "GSI-ENV-003", "machine", "CUDA_VISIBLE_DEVICES hides every GPU",
                "It is set to '', so CUDA programs started from this environment see no GPU, even though nvidia-smi lists them.",
                "Unset CUDA_VISIBLE_DEVICES, or set it to the GPU index to use (e.g. 0)."],
            ["pass", "GSI-DIFF-001", "comparison", "All compared environments have the same PyTorch build (CUDA 12.8)",
                "The difference, if any, is elsewhere; see the comparison table.", null],
        ],
    },
    ok: {
        a: CONDA_A,
        b: {
            path: "~/envs/finetune",
            python: ["Python 3.12.13", "uv venv"],
            torch: ["torch 2.11.0+cu128", "CUDA 12.8 build"],
            libs: ["15 nvidia-* wheels", "cuBLAS, cuDNN, NCCL and more"],
        },
        shell: UNSET,
        driver: DRIVER,
        gpu: GPU,
        command: "gpu-stack-inspector compare \\\n  --python ~/envs/sapiens2/bin/python \\\n  --python ~/envs/finetune/bin/python",
        findings: [
            ["pass", "GSI-DIFF-001", "comparison", "All compared environments have the same PyTorch build (CUDA 12.8)",
                "The difference, if any, is elsewhere; see the comparison table.", null],
            ["not tested", "GSI-TORCH-004", "A and B", "CUDA was not exercised; only metadata was read",
                "Everything on disk says CUDA should work. Only running something on the GPU proves it.",
                "gpu-stack-inspector probe --runtime --python ~/envs/finetune/bin/python"],
        ],
    },
};

const diagram = document.getElementById("diagram");

function fillBlock(name, layer, values) {
    const block = diagram.querySelector(`[data-block="${name}"]`);
    const [title, detail, state, code] = values;
    block.replaceChildren();
    const label = document.createElement("em");
    label.className = "block-layer";
    label.textContent = layer;
    const strong = document.createElement("strong");
    strong.textContent = title;
    const span = document.createElement("span");
    span.textContent = detail;
    block.append(label, strong, span);
    if (state) block.dataset.state = state; else delete block.dataset.state;
    if (code) block.dataset.code = code; else delete block.dataset.code;
}

function finding([status, code, subject, summary, reason, next]) {
    const item = document.createElement("li");
    item.className = "finding";
    const head = document.createElement("p");
    head.className = "finding-head";
    const badge = document.createElement("span");
    badge.className = `badge badge-${status.split(" ")[0]}`;
    badge.textContent = status.toUpperCase();
    const id = document.createElement("span");
    id.className = "finding-code";
    id.textContent = `${code} (${subject})`;
    head.append(badge, id);
    const title = document.createElement("p");
    title.className = "finding-summary";
    title.textContent = summary;
    const why = document.createElement("p");
    why.textContent = reason;
    item.append(head, title, why);
    if (next) {
        const todo = document.createElement("p");
        todo.className = "next";
        todo.textContent = next;
        item.append(todo);
    }
    return item;
}

function show(name) {
    const data = CASES[name];
    diagram.dataset.case = name;
    diagram.querySelector('[data-field="a-path"]').textContent = data.a.path;
    diagram.querySelector('[data-field="b-path"]').textContent = data.b.path;
    for (const side of ["a", "b"]) {
        fillBlock(`${side}-python`, "Python", data[side].python);
        fillBlock(`${side}-torch`, "PyTorch build", data[side].torch);
        fillBlock(`${side}-libs`, "CUDA libraries", data[side].libs);
    }
    fillBlock("shell", "Your shell", data.shell);
    fillBlock("driver", "Driver", data.driver);
    fillBlock("gpu", "GPU", data.gpu);
    diagram.querySelector('[data-field="command"]').textContent = data.command;
    diagram.querySelector('[data-field="findings"]').replaceChildren(...data.findings.map(finding));
}

const tabs = Array.from(document.querySelectorAll('.cases [role="tab"]'));

function select(tab) {
    for (const each of tabs) {
        const selected = each === tab;
        each.setAttribute("aria-selected", String(selected));
        each.tabIndex = selected ? 0 : -1;
    }
    diagram.setAttribute("aria-labelledby", tab.id);
    show(tab.dataset.case);
}

for (const tab of tabs) {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
        const index = tabs.indexOf(tab);
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
        let target = null;
        if (step) target = tabs[(index + step + tabs.length) % tabs.length];
        if (event.key === "Home") target = tabs[0];
        if (event.key === "End") target = tabs[tabs.length - 1];
        if (target) {
            event.preventDefault();
            target.focus();
            select(target);
        }
    });
}

show("cpu");

for (const button of document.querySelectorAll("[data-copy]")) {
    button.addEventListener("click", async () => {
        const text = document.getElementById(button.dataset.copy).textContent;
        try {
            await navigator.clipboard.writeText(text);
            button.textContent = "Copied";
            button.dataset.state = "done";
        } catch {
            button.textContent = "Select and copy";
        }
        setTimeout(() => {
            button.textContent = "Copy";
            delete button.dataset.state;
        }, 2000);
    });
}
