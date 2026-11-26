// Dynamic slots for "ZImage MLX LoRA Stack".
//
// The Python side declares 10 optional (lora_N, strength_N) pairs so every value
// maps to a real input. Ten empty rows are noise, so this extension hides slots
// past 2 on creation and adds a "＋" button that reveals the next pair (max 10).

import { app } from "../../scripts/app.js";

const NODE_NAME = "ZImage MLX LoRA Stack";
const MAX_SLOTS = 10;
const DEFAULT_SLOTS = 2;
const ROW_HEIGHT = 24;
const ADD_LABEL = "＋ 添加 LoRA 槽";

const find = (node, name) => node.widgets?.find((w) => w.name === name);
const slot = (node, i) => ({ lora: find(node, `lora_${i}`), strength: find(node, `strength_${i}`) });

function hideSlot(node, i) {
  for (const w of Object.values(slot(node, i))) {
    const idx = w ? node.widgets.indexOf(w) : -1;
    if (idx >= 0) node.widgets.splice(idx, 1);
  }
}

function showSlot(node, i, choices) {
  if (slot(node, i).lora) return;
  const at = Math.max(0, node.widgets.findIndex((w) => w.name === ADD_LABEL));
  const lora = node.addWidget("combo", `lora_${i}`, "none", () => {}, { values: choices });
  const strength = node.addWidget("number", `strength_${i}`, 1.0, () => {}, {
    min: -4.0, max: 4.0, step: 0.05, precision: 2,
  });
  for (const w of [lora, strength]) {
    const idx = node.widgets.indexOf(w);
    node.widgets.splice(idx, 1);
    node.widgets.splice(at, 0, w);
  }
}

function visibleSlots(node) {
  let n = 0;
  for (let i = 1; i <= MAX_SLOTS; i++) if (slot(node, i).lora) n = i;
  return n;
}

app.registerExtension({
  name: "zimage.mlx.lora_stack.slots",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      const choices = nodeData.input?.optional?.lora_1?.[0] ?? ["none"];

      for (let i = DEFAULT_SLOTS + 1; i <= MAX_SLOTS; i++) hideSlot(node, i);

      node.addWidget("button", ADD_LABEL, null, () => {
        const n = visibleSlots(node);
        if (n >= MAX_SLOTS) {
          alert(`最多 ${MAX_SLOTS} 个槽`);
          return;
        }
        showSlot(node, n + 1, choices);
        node.setSize([node.size[0], node.size[1] + ROW_HEIGHT * 2]);
        node.setDirtyCanvas(true, true);
      });

      return r;
    };
  },
});
