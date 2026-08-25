export interface ModuleNode {
  id: string;
  key?: string;
  name: string;
  parameters: string;
  children: ModuleNode[];
  path: string[];
}

export class ModelParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelParseError";
  }
}

interface ParsedExpression {
  name: string;
  parameters: string;
  children: ParsedExpression[];
}

interface LineEntry {
  indent: number;
  key?: string;
  expression: ParsedExpression;
}

export function parseModelArchitecture(source: string): ModuleNode {
  const balance = parenthesisBalance(source);
  if (balance !== 0) {
    throw new ModelParseError(balance > 0 ? "Unclosed parentheses in model definition." : "Unexpected closing parenthesis in model definition.");
  }

  const lines = source.split(/\r?\n/);
  const entries: LineEntry[] = [];

  for (const [index, rawLine] of lines.entries()) {
    if (!rawLine.trim() || /^\)+$/.test(rawLine.trim())) {
      continue;
    }

    const indent = rawLine.match(/^\s*/)?.[0].length ?? 0;
    const content = rawLine.trim();
    const match = content.match(/^\(([^)]+)\):\s*(.+)$/);
    const key = match?.[1];
    const expressionText = match?.[2] ?? content;

    try {
      entries.push({ indent, key, expression: parseExpression(expressionText) });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown parse error";
      throw new ModelParseError(`Line ${index + 1}: ${message}`);
    }
  }

  if (!entries.length) {
    throw new ModelParseError("No module definition found.");
  }

  const rootEntry = entries[0];
  const root = toNode(rootEntry.expression, rootEntry.key, [], "0");
  const stack: Array<{ indent: number; node: ModuleNode }> = [{ indent: rootEntry.indent, node: deepestNode(root) }];

  for (let index = 1; index < entries.length; index += 1) {
    const entry = entries[index];
    while (stack.length > 1 && entry.indent <= stack.at(-1)!.indent) {
      stack.pop();
    }

    const parent = stack.at(-1)?.node;
    if (!parent) {
      throw new ModelParseError(`Line ${index + 1}: could not determine parent module.`);
    }

    const child = toNode(entry.expression, entry.key, parent.path, `${parent.id}.${parent.children.length}`);
    parent.children.push(child);
    stack.push({ indent: entry.indent, node: deepestNode(child) });
  }

  return root;
}

function parseExpression(text: string): ParsedExpression {
  const openIndex = text.indexOf("(");
  if (openIndex === -1) {
    return { name: text.trim(), parameters: "", children: [] };
  }

  const name = text.slice(0, openIndex).trim();
  if (!name) {
    throw new ModelParseError("Expected a module name before '('.");
  }

  const closeIndex = matchingParenthesis(text, openIndex);
  if (closeIndex === -1) {
    const openHeader = parseOpenHeader(text);
    if (openHeader) {
      return openHeader;
    }
    throw new ModelParseError(`Unclosed parentheses in '${text}'.`);
  }

  const trailing = text.slice(closeIndex + 1).trim();
  if (trailing) {
    throw new ModelParseError(`Unexpected text after module definition: '${trailing}'.`);
  }

  const inner = text.slice(openIndex + 1, closeIndex).trim();
  const inlineChildren = parseInlineChildren(inner);
  return {
    name,
    parameters: inlineChildren.children.length ? inlineChildren.parameters : inner,
    children: inlineChildren.children.length ? inlineChildren.children : [],
  };
}

function parseOpenHeader(text: string): ParsedExpression | undefined {
  const names: string[] = [];
  let cursor = 0;
  const token = /([A-Za-z_][A-Za-z0-9_.]*)\(/y;

  while (cursor < text.length) {
    token.lastIndex = cursor;
    const match = token.exec(text);
    if (!match) {
      return undefined;
    }
    names.push(match[1]);
    cursor = token.lastIndex;
  }

  return names.reverse().reduce<ParsedExpression | undefined>(
    (child, name) => ({ name, parameters: "", children: child ? [child] : [] }),
    undefined,
  );
}

function parseInlineChildren(inner: string): { parameters: string; children: ParsedExpression[] } {
  const nestedStart = findModuleStart(inner);
  if (nestedStart === -1) {
    return { parameters: inner, children: [] };
  }

  const before = inner.slice(0, nestedStart).trim().replace(/,$/, "").trim();
  const candidate = inner.slice(nestedStart).trim();
  try {
    return { parameters: before, children: [parseExpression(candidate)] };
  } catch {
    return { parameters: inner, children: [] };
  }
}

function findModuleStart(text: string): number {
  let depth = 0;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === "(") {
      if (depth === 0) {
        let cursor = index - 1;
        while (cursor >= 0 && /[A-Za-z0-9_.]/.test(text[cursor])) {
          cursor -= 1;
        }
        const moduleName = text.slice(cursor + 1, index);
        if (/^[A-Z][A-Za-z0-9_.]*$/.test(moduleName)) {
          return cursor + 1;
        }
      }
      depth += 1;
    } else if (character === ")") {
      depth -= 1;
    }
  }
  return -1;
}

function matchingParenthesis(text: string, openIndex: number): number {
  let depth = 0;
  for (let index = openIndex; index < text.length; index += 1) {
    if (text[index] === "(") {
      depth += 1;
    } else if (text[index] === ")") {
      depth -= 1;
      if (depth === 0) {
        return index;
      }
    }
  }
  return -1;
}

function parenthesisBalance(source: string): number {
  let balance = 0;
  for (const character of source) {
    if (character === "(") {
      balance += 1;
    } else if (character === ")") {
      balance -= 1;
    }
  }
  return balance;
}

function toNode(expression: ParsedExpression, key: string | undefined, parentPath: string[], id: string): ModuleNode {
  const label = key ? `${key}: ${expression.name}` : expression.name;
  const path = [...parentPath, label];
  return {
    id,
    key,
    name: expression.name,
    parameters: expression.parameters,
    children: expression.children.map((child, index) => toNode(child, undefined, path, `${id}.${index}`)),
    path,
  };
}

function deepestNode(node: ModuleNode): ModuleNode {
  let current = node;
  while (current.children.length === 1) {
    current = current.children[0];
  }
  return current;
}