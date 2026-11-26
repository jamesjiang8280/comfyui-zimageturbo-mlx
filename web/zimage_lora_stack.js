// Dynamic slots for "ZImage MLX LoRA Stack".
//
// The Python side declares 10 optional (lora_N, strength_N) pairs so every value
// maps to a real input. Ten empty rows are noise, so this extension hides slots
// past 2 on creation and gives the node two buttons:
//   ＋ 添加 LoRA 槽   reveal the next pair (max 10), lora row above strength row
//   － 删除最后槽     hide the highest visible pair (min 1)

import { app } from "../../scripts/app.js";

const NODE_NAME = "ZImage MLX LoRA Stack";
const MAX_SLOTS = 10;
const DEFAULT_SLOTS = 2;
const MIN_SLOTS = 1;
const ROW_HEIGHT = 24;
const BTN_ADD = "＋ 添加 LoRA 槽";
const BTN_DEL = "－ 删除最后槽";

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
  // keep rows above the buttons, in reading order: lora, then strength
  let at = node.widgets.findIndex((w) => w.name === BTN_ADD);
  if (at < 0) at = node.widgets.length;
  const delIdx = node.widgets.findIndex((w) => w.name === BTN_DEL);
  if (delIdx >= 0 && delIdx < at) at = delIdx;

  const lora = node.addWidget("combo", `lora_${i}`, "none", () => {}, { values: choices });
  const strength = node.addWidget("number", `strength_${i}`, 1.0, () => {}, {
    min: -4.0, max: 4.0, step: 0.05, precision: 2,
  });

  for (const w of [lora, strength]) {
    const idx = node.widgets.indexOf(w);
    node.widgets.splice(idx, 1);
  }
  node.widgets.splice(at, 0, lora);
  node.widgets.splice(at + 1, 0, strength);   // strength below its lora
}

function visibleSlots(node) {
  let n = 0;
  for (let i = 1; i <= MAX_SLOTS; i++) if (slot(node, i).lora) n = i;
  return n;
}

function resize(node, rows) {
  node.setSize([node.size[0], Math.max(120, node.size[1] + ROW_HEIGHT * rows)]);
  node.setDirtyCanvas(true, true);
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
      // default slots must also read lora above strength
      for (let i = 1; i <= DEFAULT_SLOTS; i++) {
        const { lora, strength } = slot(node, i);
        if (!lora || !strength) continue;
        const li = node.widgets.indexOf(lora);
        const si = node.widgets.indexOf(strength);
        if (li > si) {
          node.widgets.splice(li, 1);
          node.widgets.splice(si, 0, lora);
        }
      }

      node.addWidget("button", BTN_ADD, null, () => {
        const n = visibleSlots(node);
        if (n >= MAX_SLOTS) {
          alert(`最多 ${MAX_SLOTS} 个槽`);
          return;
        }
        showSlot(node, n + 1, choices);
        resize(node, 2);
      });

      node.addWidget("button", BTN_DEL, null, () => {
        const n = visibleSlots(node);
        if (n <= MIN_SLOTS) {
          alert(`至少保留 ${MIN_SLOTS} 个槽`);
          return;
        }
        hideSlot(node, n);
        resize(node, -2);
      });

      return r;
    };
  },
});
