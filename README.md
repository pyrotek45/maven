# maven

A fast, terminal-based spreadsheet for Linux, inspired by Vim and designed for power users and data wranglers.

---

## Features

- **Vim-like navigation**: Move with `h`, `j`, `k`, `l`, or arrow keys.
- **Visual selection**: Select rows (`V`), columns (`Ctrl+v`), or blocks (`v`).
- **Command mode**: `:` to enter commands (save, load, sort, etc).
- **Tab completion**: For file paths in `:load` and `:save` commands.
- **Popup menus**: For file suggestions and other lists.
- **Inspector bar**: Shows cell content, location, and sheet info.
- **Undo/redo**: `u`/`U` for unlimited undo/redo.
- **Clipboard**: `y` (yank), `d` (delete/cut), `p` (paste).
- **Sort**: Powerful, context-aware sorting with `:sort` or `s` in visual mode.
- **Multiple tabs**: Work on several files at once (tab bar, `Tab` to switch).
- **CSV support**: Load and save CSV files easily.
- **Help popup**: `:help` or `?` for a full key/command reference.

---

## Quick Start

```sh
git clone https://github.com/yourusername/spr.git
cd spr
cargo run --release [optional.csv]
```

- Open a CSV file: `cargo run --release myfile.csv`
- Or start with a blank sheet.

---

## Key Bindings

| Key(s)         | Action                                 |
|----------------|----------------------------------------|
| h/j/k/l, Arrows| Move cursor                            |
| V              | Visual row select                      |
| v              | Visual block select                    |
| Ctrl+v         | Visual column select                   |
| :              | Enter command mode                     |
| i              | Edit cell (insert mode)                |
| u/U            | Undo/Redo                              |
| y/d/p          | Yank/Delete/Paste                      |
| <, >           | Decrease/Increase column width         |
| Tab            | Switch tab (normal mode)               |
| s (visual)     | Sort selection (see below)             |
| Esc            | Exit mode/selection/popup              |

---

## Command Mode (`:`)

- `:w `      — Save as CSV
- `:q`             — Quit
- `:wq `     — Save and quit
- `:load <file>`   — Load CSV file
- `:save <file>`   — Save as CSV
- `:files`         — Show files in current directory
- `:help`          — Show help popup
- `:sort [flags]`  — Sort selection (see below)

Tab completion works for file paths in `:load` and `:save`.

---

## Sorting

- Enter visual mode (V, v, or Ctrl+v), select a region, then type `:sort [flags]` or just `s`.
- **Flags**:
    - `n` (numeric, default)
    - `s` (string)
    - `l` (length)
    - `<` (ascending, default)
    - `>` (descending)
    - `e` (extended: sort whole rows/columns)
- **Examples**:
    - `:sort` — Numeric ascending (default)
    - `:sort s >` — String descending
    - `:sort l <` — Length ascending
    - `:sort e n >` — Extended, numeric descending

**Sort behavior:**
- Visual Line (`V`): Sort rows by the column under the cursor.
- Visual Column (`Ctrl+v`): Sort columns by the row under the cursor.
- Visual Block (`v`): Sort block rows by the leftmost column.

---

## Tips

- Use `:help` or `?` for a full key/command reference.
- Undo/redo works for all actions.
- Use tab completion for file paths.
- Inspector bar shows current cell and sheet info.
- Multiple files: open with `:tabnew <file>`, switch with `Tab`.

---

## Contributing

Pull requests and issues are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) if available.
