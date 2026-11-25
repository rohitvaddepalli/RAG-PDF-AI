import React from 'react';

function normalizeLatexSymbols(text) {
  return text
    .replace(/\$\\rightarrow\$/g, '→')
    .replace(/\\rightarrow/g, '→')
    .replace(/\$\\leftarrow\$/g, '←')
    .replace(/\\leftarrow/g, '←')
    .replace(/\$\\leftrightarrow\$/g, '↔')
    .replace(/\\leftrightarrow/g, '↔');
}

function renderInline(text) {
  const cleaned = normalizeLatexSymbols(text);
  const tokens = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = pattern.exec(cleaned)) !== null) {
    if (match.index > lastIndex) {
      tokens.push(cleaned.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      tokens.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith('`') && token.endsWith('`')) {
      tokens.push(<code key={key++} className="inline-code">{token.slice(1, -1)}</code>);
    } else if (token.startsWith('*') && token.endsWith('*')) {
      tokens.push(<em key={key++}>{token.slice(1, -1)}</em>);
    }
    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < cleaned.length) {
    tokens.push(cleaned.slice(lastIndex));
  }
  return tokens;
}

export default function FormattedAnswer({ text }) {
  if (!text) return null;

  const lines = text.split(/\r?\n/);
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const rawLine = lines[i];
    const trimmed = rawLine.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const Tag = level <= 2 ? 'h4' : 'h5';
      blocks.push(
        <Tag key={blocks.length} className="answer-heading">
          {renderInline(headingMatch[2])}
        </Tag>
      );
      i++;
      continue;
    }

    if (/^\*\*[^*]+\*\*:?$/.test(trimmed)) {
      blocks.push(
        <h4 key={blocks.length} className="answer-heading">
          {renderInline(trimmed)}
        </h4>
      );
      i++;
      continue;
    }

    if (/^\s*([*\-•])\s+/.test(rawLine)) {
      const items = [];
      while (i < lines.length && /^\s*([*\-•])\s+/.test(lines[i])) {
        const line = lines[i];
        const indent = line.search(/\S/);
        const content = line.replace(/^\s*[*\-•]\s+/, '');
        items.push({ indent, content });
        i++;
      }
      blocks.push(
        <ul key={blocks.length} className="answer-list">
          {items.map((item, idx) => (
            <li
              key={idx}
              className={item.indent >= 2 ? 'answer-subitem' : undefined}
            >
              {renderInline(item.content)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    if (/^\s*\d+\.\s+/.test(rawLine)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        const content = lines[i].replace(/^\s*\d+\.\s+/, '');
        items.push(content);
        i++;
      }
      blocks.push(
        <ol key={blocks.length} className="answer-list answer-olist">
          {items.map((content, idx) => (
            <li key={idx}>{renderInline(content)}</li>
          ))}
        </ol>
      );
      continue;
    }

    const paraLines = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,4})\s+/.test(lines[i].trim()) &&
      !/^\*\*[^*]+\*\*:?$/.test(lines[i].trim()) &&
      !/^\s*([*\-•])\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i])
    ) {
      paraLines.push(lines[i].trim());
      i++;
    }
    blocks.push(
      <p key={blocks.length} className="answer-paragraph">
        {renderInline(paraLines.join(' '))}
      </p>
    );
  }

  return <div className="formatted-answer">{blocks}</div>;
}
