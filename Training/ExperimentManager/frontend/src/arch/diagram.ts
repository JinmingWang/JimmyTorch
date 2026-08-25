import type { ModuleNode } from "./model-parser";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const CARD_WIDTH = 140;
const CARD_HEIGHT = 62;
const COLUMN_WIDTH = 184;
const ROW_GAP = 74;
const PADDING = 24;
const PAIR_STAGGER_MS = 120;
const BLOCK_AFTER_LINE_MS = 200;

export interface DiagramOptions {
  onSelect: (node: ModuleNode) => void;
}

export class ModelDiagram {
  private readonly expandedByDepth = new Map<number, string>();
  private selectedId?: string;
  private root?: ModuleNode;
  private previousVisibleIds = new Set<string>();
  private isTransitioning = false;

  constructor(private readonly target: HTMLElement, private readonly options: DiagramOptions) {}

  setModel(root: ModuleNode): void {
    this.root = root;
    this.expandedByDepth.clear();
    this.expandedByDepth.set(0, root.id);
    this.selectedId = root.id;
    this.previousVisibleIds.clear();
    this.render(root);
    this.options.onSelect(root);
  }

  private render(root: ModuleNode): void {
    const columns = this.visibleColumns(root);
    const maxRows = Math.max(...columns.map((column) => column.length));
    const width = PADDING * 2 + columns.length * COLUMN_WIDTH;
    const height = Math.max(220, PADDING * 2 + maxRows * ROW_GAP);
    const svg = element("svg", { class: "model-canvas", viewBox: `0 0 ${width} ${height}`, width: String(width), height: String(height), role: "tree", "aria-label": "Model module hierarchy" });
    const coordinates = new Map<string, { x: number; y: number; node: ModuleNode; depth: number }>();

    columns.forEach((column, depth) => {
      column.forEach((node, index) => {
        coordinates.set(node.id, { node, depth, x: PADDING + depth * COLUMN_WIDTH, y: PADDING + index * ROW_GAP });
      });
    });

    for (let depth = 1; depth < columns.length; depth += 1) {
      const parent = columns[depth - 1].find((node) => node.id === this.expandedByDepth.get(depth - 1));
      if (!parent) continue;
      const parentPosition = coordinates.get(parent.id)!;
      columns[depth].forEach((child, index) => {
        const childPosition = coordinates.get(child.id)!;
        svg.append(this.connector(parentPosition, childPosition, !this.previousVisibleIds.has(child.id), depth, index));
      });
    }

    coordinates.forEach((position) => svg.append(this.card(position, !this.previousVisibleIds.has(position.node.id))));
    this.target.replaceChildren(svg);
    this.previousVisibleIds = new Set(coordinates.keys());
  }

  private visibleColumns(root: ModuleNode): ModuleNode[][] {
    const columns: ModuleNode[][] = [[root]];
    let current = root;
    let depth = 0;
    while (current.children.length) {
      columns.push(current.children);
      const selectedId = this.expandedByDepth.get(depth + 1);
      const next = current.children.find((child) => child.id === selectedId);
      if (!next) break;
      current = next;
      depth += 1;
    }
    return columns;
  }

  private connector(parent: Position, child: Position, isEntering: boolean, depth: number, siblingIndex: number): SVGPathElement {
    const startX = parent.x + CARD_WIDTH;
    const startY = parent.y + CARD_HEIGHT / 2;
    const endX = child.x;
    const endY = child.y + CARD_HEIGHT / 2;
    const midpoint = startX + (endX - startX) / 2;
    const delay = siblingIndex * PAIR_STAGGER_MS;
    const group = element("g", { class: `connector${isEntering ? " is-entering" : ""}`, "data-depth": String(depth), style: isEntering ? `--sequence-delay: ${delay}ms` : undefined, "aria-hidden": "true" });
    group.append(
      element("path", { d: `M ${startX} ${startY} C ${midpoint} ${startY}, ${midpoint} ${endY}, ${endX} ${endY}` }),
      element("circle", { cx: String(startX), cy: String(startY), r: "4" }),
      element("circle", { cx: String(endX), cy: String(endY), r: "4" }),
    );
    return group as unknown as SVGPathElement;
  }

  private card(position: Position, isEntering: boolean): SVGGElement {
    const { node, x, y, depth } = position;
    const hasChildren = node.children.length > 0;
    const isExpanded = this.expandedByDepth.get(depth) === node.id;
    const isSelected = this.selectedId === node.id;
    const card = element("g", {
      class: `module-card category-${moduleCategory(node.name)}${hasChildren ? " is-expandable" : ""}${isExpanded ? " is-expanded" : ""}${isSelected ? " is-selected" : ""}${isEntering ? " is-entering" : ""}`,
      transform: `translate(${x} ${y})`,
      "data-depth": String(depth),
      style: isEntering ? `--sequence-delay: ${(y - PADDING) / ROW_GAP * PAIR_STAGGER_MS + BLOCK_AFTER_LINE_MS}ms` : undefined,
      tabindex: "0",
      role: "treeitem",
      "aria-expanded": hasChildren ? String(isExpanded) : undefined,
      "aria-label": `${node.name}${hasChildren ? `, ${node.children.length} submodules` : ""}`,
    }) as SVGGElement;
    card.append(element("rect", { width: String(CARD_WIDTH), height: String(CARD_HEIGHT), rx: "6" }));
    const labelClipId = `label-clip-${node.id.replaceAll(".", "-")}`;
    const clipPath = element("clipPath", { id: labelClipId });
    clipPath.append(element("rect", { x: "10", y: "4", width: hasChildren ? "104" : "120", height: "56" }));
    card.append(clipPath);

    const variableLabel = node.key ?? (depth === 0 ? "(root)" : "(anon)");
    card.append(textElement("text", { class: "card-key", x: "12", y: "18", "clip-path": `url(#${labelClipId})` }, variableLabel));
    card.append(textElement("text", { class: `card-type${node.name.length > 14 ? " is-scrolling" : ""}`, x: "12", y: "38", "clip-path": `url(#${labelClipId})` }, node.name));
    const meta = cardMeta(node);
    if (meta) {
      card.append(textElement("text", { class: "card-meta", x: "12", y: "54", "clip-path": `url(#${labelClipId})` }, meta));
    }
    if (hasChildren) {
      card.append(textElement("text", { class: "card-toggle", x: String(CARD_WIDTH - 16), y: String(CARD_HEIGHT / 2 + 6), "text-anchor": "middle" }, isExpanded ? "−" : "+"));
    }

    const activate = (): void => this.activate(node, depth, hasChildren, isExpanded);
    card.addEventListener("click", activate);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
    return card;
  }

  private activate(node: ModuleNode, depth: number, hasChildren: boolean, isExpanded: boolean): void {
    if (this.isTransitioning) return;
    this.selectedId = node.id;
    if (hasChildren && depth > 0) {
      if (isExpanded) {
        for (const key of [...this.expandedByDepth.keys()]) if (key >= depth) this.expandedByDepth.delete(key);
        this.animateCollapse(depth);
        this.options.onSelect(node);
        return;
      } else {
        this.expandedByDepth.set(depth, node.id);
        for (const key of [...this.expandedByDepth.keys()]) if (key > depth) this.expandedByDepth.delete(key);
      }
    }
    if (this.root) this.render(this.root);
    this.options.onSelect(node);
  }

  private animateCollapse(depth: number): void {
    this.isTransitioning = true;
    this.target.querySelectorAll<SVGGElement>(`.module-card[data-depth]:not([data-depth="${depth}"])`).forEach((card) => {
      if (Number(card.dataset.depth) > depth) card.classList.add("is-leaving");
    });
    this.target.querySelectorAll<SVGGElement>(`.connector[data-depth]:not([data-depth="${depth}"])`).forEach((connector) => {
      if (Number(connector.dataset.depth) > depth) connector.classList.add("is-leaving");
    });
    window.setTimeout(() => {
      if (this.root) this.render(this.root);
      this.isTransitioning = false;
    }, 330);
  }
}

interface Position {
  x: number;
  y: number;
  node: ModuleNode;
  depth: number;
}

function element(name: string, attributes: Record<string, string | undefined>): SVGElement {
  const result = document.createElementNS(SVG_NAMESPACE, name);
  Object.entries(attributes).forEach(([key, value]) => value !== undefined && result.setAttribute(key, value));
  return result;
}

function textElement(name: string, attributes: Record<string, string | undefined>, content: string): SVGTextElement {
  const result = element(name, attributes) as SVGTextElement;
  result.textContent = content;
  return result;
}

function moduleCategory(name: string): string {
  if (name === "Sequential") return "sequential";
  if (/^(Linear|MHSA)$/.test(name)) return "linear";
  if (/Dropout/.test(name)) return "dropout";
  if (/Norm/.test(name)) return "normalization";
  if (/Conv/.test(name)) return "convolution";
  if (/^(SiLU|ReLU|GELU|Sigmoid|Tanh)$/.test(name)) return "activation";
  if (/Loss$/.test(name)) return "loss";
  if (/^(Transpose|Permute|Flatten|Unflatten|Rearrange)$/.test(name)) return "shape";
  return "custom";
}

function cardMeta(node: ModuleNode): string {
  if (node.children.length > 0) return `${node.children.length} module${node.children.length === 1 ? "" : "s"}`;
  const p = node.parameters.trim();
  if (!p) return "";
  return p.length > 22 ? `${p.slice(0, 20)}…` : p;
}