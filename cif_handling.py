import re, gzip, os, shlex
import pandas as pd
from typing import List, Dict, Tuple, Any
_DATA_RE = re.compile(r'(?m)^data_(\S+)\s*$', re.MULTILINE)
CIF_MISSING_TOKENS = {'.', '?'}
_FLOAT_RE = re.compile(r'^[+-]?(?:\d+(\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
_INT_RE = re.compile(r'^[+-]?\d+$')

def cif_safe_token(value: str) -> str:
    """
    Return a CIF-safe string token.
    Always quote atom-like tokens containing a single quote (e.g. O5') as "O5'".
    Also quote tokens containing whitespace, '#', ';', or starting with a semicolon.
    """
    s = str(value)

    # If the value contains a single quote (like O5'), we quote with double quotes
    # to avoid breaking CIF parsing.
    if "'" in s:
        return f'"{s}"'

    # If it contains spaces, #, ;, or starts with a semicolon or is empty, quote with single quotes
    if any(ch.isspace() for ch in s) or s.startswith(';') or s == '' or s.startswith('#'):
        s_escaped = s.replace("'", "''")
        return f"'{s_escaped}'"

    # Otherwise safe as-is
    return s


def read_raw_cif(path: str) -> str:
    """
    Read an mmCIF file into a single string.
    Supports uncompressed files and gzip-compressed files (.gz).
    Auto-detects gzip by filename (.gz) and by checking magic bytes as a fallback.
    """
    # Fast path: filename indicates gzip
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as fh:
            return fh.read()

    # Otherwise open normally but defensively check magic bytes in case it's compressed
    with open(path, 'rb') as fh:
        start = fh.read(2)
        fh.seek(0)
        # gzip magic bytes: 0x1f 0x8b
        if start == b'\x1f\x8b':
            # it's gzipped even though filename doesn't say so
            with gzip.open(fh, 'rt', encoding='utf-8', errors='replace') as gzfh:
                return gzfh.read()
        # not gzipped -> decode normally
        raw_bytes = fh.read()
        try:
            return raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # fallback to latin-1 if weird encoding (rare)
            return raw_bytes.decode('latin-1')
        

def split_header_blocks_footer(raw: str) -> Tuple[str, Dict[str,str], str]:
    """
    Returns (header_text, blocks_dict, footer_text)
    blocks_dict: {block_name: block_text_including leading 'data_blockname' line and following lines up to before next data_}
    """
    matches = list(_DATA_RE.finditer(raw))
    if not matches:
        # No data_ found -> entire file is header
        return raw, {}, ''
    header_start = 0
    first_data_pos = matches[0].start()
    header = raw[header_start:first_data_pos]
    blocks = {}
    for i, m in enumerate(matches):
        block_name = m.group(1)
        start = m.start()
        if i + 1 < len(matches):
            end = matches[i+1].start()
        else:
            end = len(raw)
        blocks[block_name] = raw[start:end].rstrip('\n')  # keep block text (no trailing newlines)
    # anything after last block (should be none normally) is footer, but we removed it above.
    # However if there are trailing newlines after last block we already included them; to capture a footer
    # that sits after the last data_ block, we check if the last block ends before EOF:
    last_match_end = matches[-1].end()
    # Determine footer by reconstructing from raw: if blocks covers to EOF, footer empty
    last_block_end = matches[-1].start() + len(blocks[matches[-1].group(1)])
    # Simpler: compute footer as raw after the end of the block we put into blocks
    # But we used raw[start:end] where end might be len(raw) so footer empty typical
    # We'll compute footer directly:
    last_block_text = blocks[matches[-1].group(1)]
    last_block_end_index = raw.find(last_block_text) + len(last_block_text)
    footer = raw[last_block_end_index:]
    return header, blocks, footer


def tokenize_cif_line(line: str) -> List[str]:
    """
    Use shlex-like splitting but CIF tokens allow single- or double-quoted values and unquoted tokens.
    We use shlex with posix=True to handle quotes; it will treat whitespace as delimiter.
    """
    # shlex will treat '#' as comment start if we set comments=True; CIF comments are semicolon blocks or lines starting with '#'
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ''  # do NOT treat # as comment here — some CIFs use # in a value
    tokens = list(lexer)
    return tokens


def parse_loops_from_block_with_offsets(block_text: str) -> List[Dict[str, Any]]:
    """
    Parses loop_ blocks from a single data_ block and returns a list where each item is:
      {
        'tags': [...],
        'rows': [[...], ...],
        'raw_start': int,      # char offset into block_text where 'loop_' starts
        'raw_end': int,        # char offset in block_text where this loop ends (exclusive)
        'raw_text': str        # block_text[raw_start:raw_end]
      }
    Handles standard CIF quoting and multiline ';' text fields in a practical way.
    """
    lines = block_text.splitlines(keepends=True)  # preserve newlines for offset accounting
    loops = []
    i = 0
    n = len(lines)
    char_pos = 0  # running char offset into block_text
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.lower().startswith('loop_'):
            loop_start_pos = char_pos
            # move to next line to gather tags
            i += 1
            char_pos += len(line)
            tags = []
            while i < n:
                l = lines[i]
                s = l.strip()
                if s.startswith('_'):
                    # tag may be followed by other tokens on same line; keep entire tag token
                    # take first token only as tag name
                    tag_token = s.split()[0]
                    tags.append(tag_token)
                    i += 1
                    char_pos += len(l)
                else:
                    break
            # collect tokens for rows until stop condition
            tokens: List[str] = []
            while i < n:
                l = lines[i]
                s = l.strip()
                # stop if blank line? we allow blanks to be skipped
                if s == '':
                    i += 1
                    char_pos += len(l)
                    continue
                # stop conditions
                if s.lower().startswith('loop_') or s.lower().startswith('data_') or s.startswith('_'):
                    break
                # handle semicolon text field (line starting with ';')
                if l.startswith(';'):
                    # accumulate lines until a line that begins with ';'
                    # Note: the starting semicolon line is a delimiter and not included in content
                    i += 1
                    char_pos += len(l)
                    multiline_parts = []
                    while i < n and not lines[i].startswith(';'):
                        multiline_parts.append(lines[i].rstrip('\n'))
                        char_pos += len(lines[i])
                        i += 1
                    # skip terminating semicolon line if present
                    if i < n and lines[i].startswith(';'):
                        char_pos += len(lines[i])
                        i += 1
                    # token is the joined multiline (we keep internal newlines)
                    tokens.append('\n'.join(multiline_parts))
                    continue
                # normal tokenization for the line
                toklist = tokenize_cif_line(l)
                tokens.extend(toklist)
                i += 1
                char_pos += len(l)
            # chunk tokens into rows, best-effort
            rows = []
            if len(tags) > 0:
                # graceful: use integer division to keep full rows only
                row_count = len(tokens) // len(tags)
                for r in range(row_count):
                    rows.append(tokens[r*len(tags):(r+1)*len(tags)])
            loop_end_pos = char_pos
            raw_text = block_text[loop_start_pos:loop_end_pos]
            loops.append({
                'tags': tags,
                'rows': rows,
                'raw_start': loop_start_pos,
                'raw_end': loop_end_pos,
                'raw_text': raw_text
            })
        else:
            i += 1
            char_pos += len(line)
    return loops


def loop_to_dataframe(loop: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert parsed loop (tags + rows) to a pandas DataFrame with sensible dtypes:
      - integer-like columns -> pandas nullable Int64 dtype
      - float-like columns   -> pandas nullable Float64 dtype
      - others remain object (strings)
    Missing tokens ('.' or '?') become pd.NA.
    """
    tags = loop['tags']
    rows = loop['rows']
    # Build raw DataFrame of strings first
    df = pd.DataFrame(rows, columns=tags)
    # Normalize CIF missing tokens to pd.NA
    df = df.replace(list(CIF_MISSING_TOKENS), pd.NA)

    # Try to convert each column more precisely
    for col in df.columns:
        # sample non-null string representations for quick dtype detection
        sample_vals = df[col].dropna().astype(str).head(200).tolist()
        if not sample_vals:
            # nothing to infer
            continue

        # check integer-like
        if all(_INT_RE.match(s) for s in sample_vals):
            # convert full column to numeric (coerce any weird values to NaN) and then to pandas nullable Int64
            numeric = pd.to_numeric(df[col], errors='coerce')
            try:
                df[col] = numeric.astype('Int64')  # nullable integer type
            except Exception:
                # fallback to numeric float if astype fails for any reason
                df[col] = numeric.astype('Float64')
            continue

        # check float-like (includes integer-like strings but we've already filtered ints)
        if all(_FLOAT_RE.match(s) for s in sample_vals):
            numeric = pd.to_numeric(df[col], errors='coerce')
            # use pandas nullable float dtype
            df[col] = numeric.astype('Float64')
            continue

        # otherwise: keep as string/object (but still keep pd.NA preserved)
        # ensure pure strings (no accidental numeric types)
        df[col] = df[col].astype(object)

    return df


def infer_start_columns(loop_info: Dict[str, Any]) -> Dict[str, int]:
    """
    Inspect the raw_text of a parsed loop to find the column starting positions for each tag.
    Returns mapping { tag : 1-based start column }.
    Strategy:
      - look at the textual lines of loop_info['raw_text']
      - skip the header lines (loop_ and subsequent tag lines)
      - for each data line, tokenise using tokenize_cif_line(), then use line.find(token, pos) to detect start offsets
      - collect minimum offset seen for each column across rows (robust if some rows shorter)
    """
    raw = loop_info['raw_text']
    lines = raw.splitlines()
    # skip initial 'loop_' and tag lines: find first data-line index
    data_lines = []
    i = 0
    # skip lines that are 'loop_' or tags (start with '_')
    while i < len(lines):
        s = lines[i].strip()
        if s.lower().startswith('loop_'):
            i += 1
            continue
        if s.startswith('_'):
            i += 1
            continue
        # first non-tag non-loop line begins data region
        break
    # collect subsequent lines until next blank / or end or semicolon-blocks
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        # stop at blank line or comment/ new block start or a tag starting line (defensive)
        if s == '':
            i += 1
            continue
        if s.lower().startswith('loop_') or s.lower().startswith('data_') or s.startswith('_'):
            break
        data_lines.append(line)
        i += 1

    if not data_lines:
        return {t: 1 for t in loop_info['tags']}  # fallback

    tag_starts: Dict[str, List[int]] = {t: [] for t in loop_info['tags']}

    for line in data_lines:
        # skip semicolon block lines (those starting with ';' and not data)
        if line.startswith(';'):
            # semicolon content lines are not standard token lines
            continue
        # tokenise this textual data line using the tokenizer that handles quotes
        toks = tokenize_cif_line(line)
        if not toks:
            continue
        search_pos = 0
        # for robustness: attempt to map tokens in order to tags
        for idx, tok in enumerate(toks):
            # token may be quoted in the raw line; find it with a substring search starting at search_pos
            # prefer exact token match; try both quoted and unquoted forms
            candidates = [tok]
            # if token contains single-quote, token in raw may be quoted; try quoted forms
            candidates.append("'" + tok.replace("'", "''") + "'")
            candidates.append(f'"{tok}"')
            found = -1
            for cand in candidates:
                found = line.find(cand, search_pos)
                if found != -1:
                    break
            if found == -1:
                # fallback: find bare token
                found = line.find(tok.split()[0], search_pos)
            if found == -1:
                # give up mapping this token for this line
                # but continue with approximate spacing: estimate position using search_pos
                pos = search_pos
            else:
                pos = found
            # matched token belongs to column idx (if idx < number of tags)
            if idx < len(loop_info['tags']):
                tag = loop_info['tags'][idx]
                tag_starts[tag].append(pos + 1)  # convert to 1-based column
            # move search_pos just after found token to find subsequent tokens correctly
            search_pos = pos + max(len(tok), 1)

    # compute final start column as the minimum observed start for each tag (or 1 if none)
    final_starts: Dict[str, int] = {}
    for tag, starts in tag_starts.items():
        if starts:
            final_starts[tag] = min(starts)
        else:
            final_starts[tag] = 1
    return final_starts


def infer_decimal_places(loop_info: Dict[str, Any], df: pd.DataFrame) -> Dict[str, int]:
    """
    Inspect original token strings in loop_info['rows'] and determine decimal places per column.
    Rules:
      - If column is one of Cartn_x/y/z -> enforce 3 decimals
      - Else if any observed decimal places > 0, use the max observed (to match input)
      - Else default to 2 decimals for floats
      - For non-floats, value is ignored
    Returns mapping { tag : decimals } for float-like columns only.
    """
    tags = loop_info['tags']
    decimals: Dict[str, int] = {}
    # iterate columns
    for col_idx, tag in enumerate(tags):
        # only consider columns that are float-like in df or whose sample matches float regex
        if col_idx >= len(loop_info.get('rows', [])) and not pd.api.types.is_float_dtype(df.get(tag, pd.Series()).dtype):
            # if no row tokens and not float dtype, skip
            continue
        # gather token strings by reading loop_info['rows'] (these are original token strings)
        tokens = [row[col_idx] for row in loop_info.get('rows', []) if len(row) > col_idx]
        token_strs = [str(t) for t in tokens if isinstance(t, (str, int, float))]
        max_dec = 0
        for ts in token_strs:
            # skip missing tokens
            if ts in ('.', '?'):
                continue
            # strip quotes if present
            unq = ts
            if (unq.startswith("'") and unq.endswith("'")) or (unq.startswith('"') and unq.endswith('"')):
                unq = unq[1:-1]
            # match float
            m = re.match(r'^[+-]?(\d+)\.(\d+)(?:[eE][+-]?\d+)?$', unq)
            if m:
                decs = len(m.group(2))
                if decs > max_dec:
                    max_dec = decs
            else:
                # scientific notation like 1e-3 -> treat decimals as default later
                pass
        # special-case: Cartn columns => 3 decimals regardless
        if any(k in tag for k in ('Cartn_x', 'Cartn_y', 'Cartn_z')):
            decimals[tag] = 3
            continue
        if max_dec > 0:
            decimals[tag] = max_dec
        else:
            # fallback default for floats: 2
            decimals[tag] = 2
    return decimals


def write_loop_from_df_aligned(df: pd.DataFrame,
                               loop_info: Dict[str, Any],
                               tag_order: List[str] = None,
                               start_cols: Dict[str, int] = None,
                               decimals_map: Dict[str, int] = None,
                               float_fmt_template: str = None,
                               missing_token: str = '.',
                               indent: str = '') -> str:
    """
    Write a CIF loop with fixed column start positions and decimal precision matched to input.
    Automatically appends a trailing '#' line for atom_site loops (contains '_atom_site.').
    """
    if tag_order is None:
        tag_order = loop_info.get('tags', list(df.columns))
    # infer starts and decimals if not provided
    if start_cols is None:
        start_cols = infer_start_columns(loop_info)
    if decimals_map is None:
        decimals_map = infer_decimal_places(loop_info, df)

    # detect atom_site loop
    is_atom_loop = any('_atom_site.' in t for t in tag_order)

    # Build header
    lines_out = []
    lines_out.append(f'{indent}loop_')
    for t in tag_order:
        lines_out.append(f'{indent}{t}')

    # prepare type info
    is_int_col = {col: pd.api.types.is_integer_dtype(df[col].dtype) for col in tag_order}
    is_float_col = {col: pd.api.types.is_float_dtype(df[col].dtype) for col in tag_order}

    max_start = max((start_cols.get(t, 1) for t in tag_order), default=1)
    line_base_len = max_start + 20

    for _, row in df.iterrows():
        charlist = [' '] * (line_base_len)
        def ensure_len(L, needed):
            if needed > len(L):
                L.extend([' '] * (needed - len(L)))

        multiline_cells = []  # (start_col, tag, content)

        for tag in tag_order:
            start_col = start_cols.get(tag, 1)
            start_idx = max(0, start_col - 1)
            v = row.get(tag, pd.NA)
            if pd.isna(v):
                token = missing_token
            else:
                if is_int_col.get(tag, False):
                    try:
                        token = str(int(v))
                    except Exception:
                        token = str(v)
                elif is_float_col.get(tag, False):
                    p = decimals_map.get(tag, 2)
                    if float_fmt_template:
                        try:
                            token = float_fmt_template.format(float(v), p=p)
                        except Exception:
                            token = ('{0:.' + str(p) + 'f}').format(float(v))
                    else:
                        token = ('{0:.' + str(p) + 'f}').format(float(v))
                else:
                    s = str(v)
                    if '\n' in s:
                        token = missing_token
                        multiline_cells.append((start_idx + 1, tag, s))
                    else:
                        token = cif_safe_token(s)
            needed_len = start_idx + len(token)
            ensure_len(charlist, needed_len)
            for k, ch in enumerate(token):
                charlist[start_idx + k] = ch

        row_line = ''.join(charlist).rstrip()
        lines_out.append(indent + row_line)

        # multiline blocks
        for start_col1, tag, content in multiline_cells:
            lines_out.append(indent + ';')
            content_str = content
            if content_str.startswith('\n'):
                content_str = content_str[1:]
            for content_line in content_str.splitlines():
                lines_out.append(content_line)
            lines_out.append(indent + ';')

    # Append trailing '#' for atom_site loops
    if is_atom_loop:
        lines_out.append('#')

    return '\n'.join(lines_out) + '\n'


def replace_loop_in_block_text(block_text: str, loop_info: Dict[str, Any], new_loop_text: str) -> str:
    """
    Given the block_text and a parsed loop_info (with raw_start/raw_end),
    return a new block_text with that substring replaced by new_loop_text.
    """
    start = loop_info['raw_start']
    end = loop_info['raw_end']
    new_block = block_text[:start] + new_loop_text + block_text[end:]
    return new_block


def write_cif_from_parts(header: str, blocks: Dict[str, str], footer: str, outpath: str):
    """
    Write CIF composed of header + blocks + footer to outpath.
    If outpath ends with .gz it will be gzip-compressed.
    """
    # Build full text in memory (small to moderate files are fine). If files can be huge,
    # you can stream directly to gzip.open / open instead.
    parts = []
    if header:
        parts.append(header)
    for bname, btext in blocks.items():
        # ensure a single trailing newline between blocks
        parts.append(btext.rstrip('\n') + '\n\n')
    if footer:
        parts.append(footer)
    full_text = ''.join(parts)

    # write compressed if requested
    if outpath.endswith('.gz'):
        # ensure directory exists
        os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
        with gzip.open(outpath, 'wt', encoding='utf-8') as fh:
            fh.write(full_text)
    else:
        os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
        with open(outpath, 'w', encoding='utf-8') as fh:
            fh.write(full_text)


def canonical_atom_site_order() -> List[str]:
    """
    Example canonical order for atom_site tags (extend as you like).
    This is the order you want to save as standard.
    """
    return [
        '_atom_site.group_PDB',
        '_atom_site.id',
        '_atom_site.type_symbol',
        '_atom_site.label_atom_id',
        '_atom_site.label_alt_id',
        '_atom_site.label_comp_id',
        '_atom_site.label_asym_id',
        '_atom_site.label_entity_id',
        '_atom_site.label_seq_id',
        '_atom_site.pdbx_PDB_ins_code',
        '_atom_site.Cartn_x',
        '_atom_site.Cartn_y',
        '_atom_site.Cartn_z',
        '_atom_site.occupancy',
        '_atom_site.B_iso_or_equiv',
        '_atom_site.pdbx_formal_charge',
        '_atom_site.auth_seq_id',
        '_atom_site.auth_comp_id',
        '_atom_site.auth_asym_id',
        '_atom_site.auth_atom_id',
        '_atom_site.pdbx_PDB_model_num'
    ]


def reorder_df_to_canonical(df: pd.DataFrame, canonical_order: List[str]) -> pd.DataFrame:
    """
    Return a DataFrame with columns reordered according to canonical_order.
    Any columns present in canonical_order but missing in df will be created filled with pd.NA.
    Any extra columns present in df but not in canonical_order are appended (in original order).
    """
    # columns to ensure present in result (preserve df columns not mentioned)
    extra_cols = [c for c in df.columns if c not in canonical_order]
    # build final order: all canonical + extras
    desired = [c for c in canonical_order] + extra_cols
    # reindex will add missing columns with NaN (pd.NA preserved by dtype semantics)
    return df.reindex(columns=desired)


