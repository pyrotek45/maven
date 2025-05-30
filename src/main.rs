// main.rs
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style}, // Keep Style, Color, Modifier as they are used in PopupMenu::draw indirectly
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph}, // Keep List, ListItem as they are used in PopupMenu::draw indirectly
};
use std::{
    collections::{HashMap, VecDeque, HashSet},
    env,
    fs,
    fs::File,
    io,
    path::PathBuf, // Removed unused Path
    time::Duration,
}; // Added env

#[derive(Clone, Debug)]
pub enum CellValue {
    Text(String),
    Number(f64),
    Formula(String), // Lua expression, e.g. "A1 + B2"
}

#[derive(Clone)]
struct Cell {
    value: CellValue,
}

impl Cell {
    pub fn from_content(content: &str) -> Self {
        if let Some(rest) = content.strip_prefix('=') {
            Cell { value: CellValue::Formula(rest.to_string()) }
        } else if let Ok(n) = content.parse::<f64>() {
            Cell { value: CellValue::Number(n) }
        } else {
            Cell { value: CellValue::Text(content.to_string()) }
        }
    }
    pub fn to_content(&self) -> String {
        match &self.value {
            CellValue::Text(s) => s.clone(),
            CellValue::Number(n) => n.to_string(),
            CellValue::Formula(expr) => format!("={}", expr),
        }
    }
}

// Convert (row, col) to Excel-style name (A1, B2, etc.)
pub fn cell_name(row: usize, col: usize) -> String {
    let mut col_str = String::new();
    let mut col_num = col + 1;
    while col_num > 0 {
        let rem = (col_num - 1) % 26;
        col_str.insert(0, (b'A' + rem as u8) as char);
        col_num = (col_num - 1) / 26;
    }
    format!("{}{}", col_str, row + 1)
}

// Convert Excel-style name to (row, col)
pub fn parse_cell_name(name: &str) -> Option<(usize, usize)> {
    let mut col = 0;
    let mut row = 0;
    let mut chars = name.chars();
    // Parse column (letters)
    while let Some(c) = chars.clone().next() {
        if c.is_ascii_alphabetic() {
            col = col * 26 + ((c.to_ascii_uppercase() as u8 - b'A') as usize + 1);
            chars.next();
        } else {
            break;
        }
    }
    // Parse row (numbers)
    let row_str: String = chars.collect();
    if row_str.is_empty() {
        return None;
    }
    row = row_str.parse::<usize>().ok()?;
    Some((row - 1, col - 1))
}

// --- Remove Lua support: delete use mlua, SpreadsheetContext, evaluate_formula, and Lua usage ---

// --- Enhanced formula evaluator: support SUM and ranges ---
#[derive(Debug, Clone)]
enum Expr {
    Number(f64),
    CellRef(usize, usize),
    Range((usize, usize), (usize, usize)),
    BinaryOp(Box<Expr>, Op, Box<Expr>),
    FuncCall(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

// Parse a cell reference like "A0" into (row, col), where both are zero-based.
// E.g. "A0" -> (0, 0), "B2" -> (2, 1), "AA0" -> (0, 26)
fn parse_cell_ref(s: &str) -> Option<(usize, usize)> {
    let mut col = 0;
    let mut chars = s.chars().peekable();
    let mut col_len = 0;
    while let Some(&c) = chars.peek() {
        match c {
            'A'..='Z' | 'a'..='z' => {
                col = col * 26 + ((c.to_ascii_uppercase() as u8 - b'A') as usize);
                chars.next();
                col_len += 1;
            }
            _ => break,
        }
    }
    if col_len == 0 {
        return None;
    }
    let row_str: String = chars.collect();
    if row_str.is_empty() {
        return None;
    }
    let row = row_str.parse::<usize>().ok()?;
    Some((row, col))
}

// A very simple recursive descent parser for +, -, *, /, (), and cell refs
fn parse_expr(input: &str) -> Option<Expr> {
    let tokens = tokenize(input);
    let (expr, rest) = parse_expr_bp(&tokens, 0)?;
    if rest.is_empty() { Some(expr) } else { None }
}

fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            '0'..='9' | '.' => {
                let mut num = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() || d == '.' {
                        num.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if let Ok(n) = num.parse() {
                    tokens.push(Token::Number(n));
                }
            }
            'A'..='Z' | 'a'..='z' => {
                let mut ident = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_alphabetic() || d.is_ascii_digit() {
                        ident.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Ident(ident));
            }
            ':' => { tokens.push(Token::Colon); chars.next(); }
            ',' => { tokens.push(Token::Comma); chars.next(); }
            '+' => { tokens.push(Token::Plus); chars.next(); }
            '-' => { tokens.push(Token::Minus); chars.next(); }
            '*' => { tokens.push(Token::Star); chars.next(); }
            '/' => { tokens.push(Token::Slash); chars.next(); }
            '(' => { tokens.push(Token::LParen); chars.next(); }
            ')' => { tokens.push(Token::RParen); chars.next(); }
            ' ' | '\t' => { chars.next(); }
            _ => { chars.next(); }
        }
    }
    tokens
}

#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Ident(String),
    Colon,
    Comma,
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

// Pratt parser with support for ranges and function calls
fn parse_expr_bp(tokens: &[Token], min_bp: u8) -> Option<(Expr, &[Token])> {
    let (mut lhs, mut rest) = match tokens.first()? {
        Token::Number(n) => (Expr::Number(*n), &tokens[1..]),
        Token::Ident(s) => {
            // Function call or cell ref or range
            if let Some(Token::LParen) = tokens.get(1) {
                // Function call
                let mut args = Vec::new();
                let mut rem = &tokens[2..];
                if let Some(Token::RParen) = rem.first() {
                    rem = &rem[1..];
                } else {
                    loop {
                        let (arg, new_rem) = parse_expr_bp(rem, 0)?;
                        args.push(arg);
                        rem = new_rem;
                        match rem.first() {
                            Some(Token::Comma) => rem = &rem[1..],
                            Some(Token::RParen) => { rem = &rem[1..]; break; },
                            _ => return None,
                        }
                    }
                }
                (Expr::FuncCall(s.to_ascii_uppercase(), args), rem)
            } else if let Some(Token::Colon) = tokens.get(1) {
                // Range: e.g. B7:D7
                if let (Some((r1, c1)), Some(Token::Ident(s2))) = (parse_cell_ref(s), tokens.get(2)) {
                    if let Some((r2, c2)) = parse_cell_ref(s2) {
                        (Expr::Range((r1, c1), (r2, c2)), &tokens[3..])
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            } else {
                // Cell ref
                if let Some((r, c)) = parse_cell_ref(s) {
                    (Expr::CellRef(r, c), &tokens[1..])
                } else {
                    return None;
                }
            }
        }
        Token::Minus => {
            let (rhs, rest) = parse_expr_bp(&tokens[1..], 100)?;
            (Expr::BinaryOp(Box::new(Expr::Number(0.0)), Op::Sub, Box::new(rhs)), rest)
        }
        Token::LParen => {
            let (expr, rest) = parse_expr_bp(&tokens[1..], 0)?;
            if let Some(Token::RParen) = rest.first() {
                (expr, &rest[1..])
            } else {
                return None;
            }
        }
        _ => return None,
    };
    loop {
        let op = match rest.first() {
            Some(Token::Plus) => Op::Add,
            Some(Token::Minus) => Op::Sub,
            Some(Token::Star) => Op::Mul,
            Some(Token::Slash) => Op::Div,
            _ => break,
        };
        let (l_bp, r_bp) = match op {
            Op::Add | Op::Sub => (1, 2),
            Op::Mul | Op::Div => (3, 4),
        };
        if l_bp < min_bp { break; }
        rest = &rest[1..];
        let (rhs, new_rest) = parse_expr_bp(rest, r_bp)?;
        lhs = Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs));
        rest = new_rest;
    }
    Some((lhs, rest))
}

// Evaluate the expression, using a cell map for lookups
fn eval_expr(
    expr: &Expr,
    cell_map: &HashMap<(usize, usize), &CellValue>,
    visited: &mut HashSet<(usize, usize)>
) -> f64 {
    match expr {
        Expr::Number(n) => *n,
        Expr::CellRef(r, c) => {
            // Detect circular reference
            if !visited.insert((*r, *c)) {
                // Already visited: circular reference!
                return f64::NAN;
            }
            let result = match cell_map.get(&(*r, *c)) {
                Some(CellValue::Number(n)) => *n,
                Some(CellValue::Formula(expr_str)) => {
                    if let Some(expr) = parse_expr(expr_str) {
                        let val = eval_expr(&expr, cell_map, visited);
                        val
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };
            // Only remove if this was the top-level call for this cell
            visited.remove(&(*r, *c));
            result
        }
        Expr::Range((r1, c1), (r2, c2)) => {
            let (rmin, rmax) = (r1.min(r2), r1.max(r2));
            let (cmin, cmax) = (c1.min(c2), c1.max(c2));
            let mut sum = 0.0;
            for r in *rmin..=*rmax {
                for c in *cmin..=*cmax {
                    // If already visited, treat as circular reference for this cell
                    if visited.contains(&(r, c)) {
                        sum += f64::NAN;
                        continue;
                    }
                    if let Some(CellValue::Number(n)) = cell_map.get(&(r, c)) {
                        sum += *n;
                    } else if let Some(CellValue::Formula(expr_str)) = cell_map.get(&(r, c)) {
                        if let Some(expr) = parse_expr(expr_str) {
                            visited.insert((r, c));
                            let val = eval_expr(&expr, cell_map, visited);
                            visited.remove(&(r, c));
                            sum += val;
                        }
                    }
                }
            }
            sum
        }
        Expr::BinaryOp(lhs, op, rhs) => {
            let a = eval_expr(lhs, cell_map, visited);
            let b = eval_expr(rhs, cell_map, visited);
            match op {
                Op::Add => a + b,
                Op::Sub => a - b,
                Op::Mul => a * b,
                Op::Div => if b == 0.0 { 0.0 } else { a / b },
            }
        }
        Expr::FuncCall(name, args) => {
            match name.as_str() {
                "SUM" => args.iter().map(|e| eval_expr(e, cell_map, visited)).sum(),
                "AVERAGE" => {
                    let sum: f64 = args.iter().map(|e| eval_expr(e, cell_map, visited)).sum();
                    let count = args.len() as f64;
                    if count > 0.0 { sum / count } else { 0.0 }
                }
                "COUNT" => args.iter().filter_map(|e| {
                    let val = eval_expr(e, cell_map, visited);
                    if val.is_finite() { Some(1.0) } else { None }
                }).sum(),
                "MAX" => args.iter().map(|e| eval_expr(e, cell_map, visited)).fold(f64::NEG_INFINITY, f64::max),
                "MIN" => args.iter().map(|e| eval_expr(e, cell_map, visited)).fold(f64::INFINITY, f64::min),
                _ => 0.0,
            }
        }
    }
}
//  --- End native formula evaluator ---

#[derive(Debug, PartialEq, Clone)] // Added Debug
enum Mode {
    Normal,
    Insert,
    Command,
    VisualRow,
    VisualColumn,
    VisualBlock,
}

#[derive(Clone)]
struct AppSnapshot {
    cells: Vec<Vec<Cell>>,
}

struct App {
    rows: usize,
    cols: usize,
    cwidth: usize,
    cells: Vec<Vec<Cell>>,
    cursor_row: usize,
    cursor_col: usize,
    scroll_row: usize,
    scroll_col: usize,
    view_rows: usize,
    view_cols: usize,
    mode: Mode,
    input: String,
    command_msg: String,
    clipboard: Vec<Vec<Cell>>,
    snapshots: VecDeque<AppSnapshot>,
    redo_stack: VecDeque<AppSnapshot>,
    visual_start: Option<(usize, usize)>,
    visual_mode_for_command: Option<Mode>, // Added field
    command_history: Vec<String>,          // For command history
    command_history_index: Option<usize>,  // Current position in history
    current_command_input_buffer: String,  // Buffer for current input when navigating history
    help_popup: Option<HelpPopup>,         // To store help popup state
    help_text: Option<String>,             // To store help message content

    // For generic popup menu
    popup_menu: Option<PopupMenu>,
    // Context for tab completion when popup_menu is active
    tab_completion_command_prefix: String, // e.g., "load " or "save "
    tab_completion_path_prefix: String,    // e.g., "some/directory/"
    filepath: Option<PathBuf>,             // Store the path of the currently loaded file

    last_selection_range: Option<((usize, usize), (usize, usize))>, // Last selected range for commands
    insert_mode_cursor: usize,
    result: Option<f64>
}

impl App {
    // delete row and column methods
    /// Delete a row at the specified index.
    pub fn delete_row(&mut self, index: usize) {
        if index < self.rows {
            self.cells.remove(index);
            self.rows = self.rows.saturating_sub(1);
        }
    }

    /// Delete a column at the specified index.
    pub fn delete_col(&mut self, index: usize) {
        if index < self.cols {
            for row in &mut self.cells {
                row.remove(index);
            }
            self.cols = self.cols.saturating_sub(1);
        }
    }


    /// Insert a new row at the specified index.
    pub fn insert_row(&mut self, index: usize) {
        let idx = index.min(self.rows);
        let new_row = vec![
            Cell {
                value: CellValue::Text("".to_string())
            };
            self.cols
        ];
        self.cells.insert(idx, new_row);
        self.rows += 1;
    }

    /// Insert a new column at the specified index.
    pub fn insert_col(&mut self, index: usize) {
        let idx = index.min(self.cols);
        for row in &mut self.cells {
            row.insert(idx, Cell { value: CellValue::Text("".to_string()) });
        }
        self.cols += 1;
    }

    pub fn new(rows: usize, cols: usize) -> Self {
        let cwidth = 10;
        let cells = vec![
            vec![
                Cell {
                    value: CellValue::Text("".to_string())
                };
                cols
            ];
            rows
        ];
        App {
            rows,
            cols,
            cwidth,
            cells,
            cursor_row: 0,
            cursor_col: 0,
            scroll_row: 0,
            scroll_col: 0,
            view_rows: 0,
            view_cols: 0,
            mode: Mode::Normal,
            input: String::new(),
            command_msg: String::new(),
            clipboard: Vec::new(),
            snapshots: VecDeque::new(),
            redo_stack: VecDeque::new(),
            visual_start: None,
            visual_mode_for_command: None,
            command_history: Vec::new(),
            command_history_index: None,
            current_command_input_buffer: String::new(),
            help_popup: None,
            help_text: None,
            popup_menu: None,
            tab_completion_command_prefix: String::new(),
            tab_completion_path_prefix: String::new(),
            filepath: None,
            last_selection_range: None, // Initialize to None
            insert_mode_cursor: 0, // Initialize insert mode cursor position
            result:None, // Initialize result to 0.0
        }
    }
    /// Take a snapshot of the current state for undo/redo.
    fn snapshot(&mut self) {
        let snap = AppSnapshot {
            cells: self.cells.clone(),
        };
        self.snapshots.push_back(snap);
        // Limit the number of snapshots to avoid unbounded memory growth
        while self.snapshots.len() > 100 {
            self.snapshots.pop_front();
        }
        // Clear redo stack on new action
        self.redo_stack.clear();
    }

    fn undo(&mut self) {
        if let Some(last_snapshot) = self.snapshots.pop_back() {
            // Save current state to redo stack
            let current = AppSnapshot {
                cells: self.cells.clone(),
            };
            self.redo_stack.push_back(current);
            self.cells = last_snapshot.cells;
        }
    }

    fn redo(&mut self) {
        if let Some(next_snapshot) = self.redo_stack.pop_back() {
            // Save current state to undo stack
            let current = AppSnapshot {
                cells: self.cells.clone(),
            };
            self.snapshots.push_back(current);
            self.cells = next_snapshot.cells;
        }
    }

    /// Returns the selected range as ((r1, c1), (r2, c2)), or None if no selection.
    /// The interpretation depends on the current visual mode.
    fn selected_range(&self) -> Option<((usize, usize), (usize, usize))> {
        let (start_row, start_col) = self.visual_start?;
        let (end_row, end_col) = (self.cursor_row, self.cursor_col);

        match self.mode {
            Mode::VisualRow => {
                let r1 = start_row.min(end_row);
                let r2 = start_row.max(end_row);
                Some(((r1, 0), (r2, self.cols.saturating_sub(1))))
            }
            Mode::VisualColumn => {
                let c1 = start_col.min(end_col);
                let c2 = start_col.max(end_col);
                Some(((0, c1), (self.rows.saturating_sub(1), c2)))
            }
            Mode::VisualBlock => {
                let r1 = start_row.min(end_row);
                let r2 = start_row.max(end_row);
                let c1 = start_col.min(end_col);
                let c2 = start_col.max(end_col);
                Some(((r1, c1), (r2, c2)))
            }
            // For other modes, treat as block selection if visual_start is set
            _ => {
                match self.visual_mode_for_command {
                    Some(Mode::VisualRow) => {
                        let r1 = start_row.min(end_row);
                        let r2 = start_row.max(end_row);
                        Some(((r1, 0), (r2, self.cols.saturating_sub(1))))
                    }
                    Some(Mode::VisualColumn) => {
                        let c1 = start_col.min(end_col);
                        let c2 = start_col.max(end_col);
                        Some(((0, c1), (self.rows.saturating_sub(1), c2)))
                    }
                    Some(Mode::VisualBlock) => {
                        let r1 = start_row.min(end_row);
                        let r2 = start_row.max(end_row);
                        let c1 = start_col.min(end_col);
                        let c2 = start_col.max(end_col);
                        Some(((r1, c1), (r2, c2)))
                    }
                    _ => None, // No selection or not in visual mode
                }
            }
        }
    }
}

// --- Sort Command Structures ---
#[derive(Debug, Clone, Copy, PartialEq)]
enum SortType {
    Numeric,
    String,
    Length,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SortOrder {
    Ascending,
    Descending,
}

#[derive(Debug, Clone)]
struct SortParameters {
    sort_type: SortType,
    sort_order: SortOrder,
    extended: bool,
}

impl Default for SortParameters {
    fn default() -> Self {
        SortParameters {
            sort_type: SortType::Numeric,
            sort_order: SortOrder::Ascending,
            extended: false,
        }
    }
}

fn parse_sort_flags(args: &[&str]) -> Result<SortParameters, String> {
    let mut params = SortParameters::default();
    let mut type_specified = false;
    let mut order_specified = false;

    for arg in args {
        match arg.to_lowercase().as_str() {
            "s" | "str" | "string" => {
                if type_specified {
                    return Err("Sort type specified multiple times".to_string());
                }
                params.sort_type = SortType::String;
                type_specified = true;
            }
            "n" | "num" | "numeric" => {
                if type_specified {
                    return Err("Sort type specified multiple times".to_string());
                }
                params.sort_type = SortType::Numeric;
                type_specified = true;
            }
            "l" | "len" | "length" => {
                if type_specified {
                    return Err("Sort type specified multiple times".to_string());
                }
                params.sort_type = SortType::Length;
                type_specified = true;
            }
            ">" | "desc" => {
                if order_specified {
                    return Err("Sort order specified multiple times".to_string());
                }
                params.sort_order = SortOrder::Descending;
                order_specified = true;
            }
            "<" | "asc" => {
                if order_specified {
                    return Err("Sort order specified multiple times".to_string());
                }
                params.sort_order = SortOrder::Ascending;
                order_specified = true;
            }
            "e" | "ext" | "extended" => {
                params.extended = true;
            }
            _ => return Err(format!("Unknown sort flag: {}", arg)),
        }
    }
    Ok(params)
}

// Core sort execution logic
fn execute_sort(
    app: &mut App,
    visual_mode: Mode, // The visual mode active when :sort was called
    mut r1: usize,
    mut c1: usize,
    mut r2: usize,
    mut c2: usize, // The selection bounds from app.selected_range()
    params: SortParameters,
) -> Result<String, String> {
    app.snapshot(); // Always take a snapshot before sorting

    match visual_mode {
        Mode::VisualRow => {
            // sort rows by selected columns, using the first row of the selection as the key
            // Default to cursor row if selection doesn't define a specific row range
            let sort_key_row_index = r1; // r1 is the start row of the selection
            // use all columns, since we are sorting by row
            c1 = 0; // Start from the first column
            c2 = app.cols.saturating_sub(1); // End at the last column
            r1 = app
                .visual_start
                .unwrap_or((app.cursor_row, app.cursor_col))
                .0
                .min(app.cursor_row);
            r2 = app
                .visual_start
                .unwrap_or((app.cursor_row, app.cursor_col))
                .0
                .max(app.cursor_row);
            let mut row_indices_to_sort: Vec<usize> = (r1..=r2).collect();
            row_indices_to_sort.sort_by_cached_key(|&row_idx| {
                app.cells[row_idx][sort_key_row_index.min(app.cols.saturating_sub(1))]
                    .to_content()
                    .clone()
            });
            if params.sort_order == SortOrder::Descending {
                row_indices_to_sort.reverse();
            }
            row_indices_to_sort.sort_by(|&row_a_idx, &row_b_idx| {
                let val_a = &app.cells[row_a_idx]
                    [sort_key_row_index.min(app.cols.saturating_sub(1))]
                .to_content();
                let val_b = &app.cells[row_b_idx]
                    [sort_key_row_index.min(app.cols.saturating_sub(1))]
                .to_content();
                compare_values(val_a, val_b, params.sort_type, params.sort_order)
            });
            let mut new_cells_state = app.cells.clone();
            if params.extended {
                // Sort entire application rows based on the key row from the selection
                for (new_relative_idx, &original_row_idx) in row_indices_to_sort.iter().enumerate()
                {
                    new_cells_state[r1 + new_relative_idx] = app.cells[original_row_idx].clone();
                }
            } else {
                // Sort rows only within the column bounds (c1..=c2) of the selection
                let original_block_data: Vec<Vec<Cell>> = row_indices_to_sort
                    .iter()
                    .map(|&orig_row_idx| {
                        (c1..=c2)
                            .map(|col_idx| app.cells[orig_row_idx][col_idx].clone())
                            .collect()
                    })
                    .collect();

                for (new_relative_idx, sorted_row_data_for_block) in
                    original_block_data.iter().enumerate()
                {
                    for (col_offset, cell_data) in sorted_row_data_for_block.iter().enumerate() {
                        new_cells_state[r1 + new_relative_idx][c1 + col_offset] = cell_data.clone();
                    }
                }
            }
            app.cells = new_cells_state;
            Ok(format!(
                "Sorted columns {}-{} by row {} ({:?}, {:?}, ext={})",
                c1, c2, sort_key_row_index, params.sort_type, params.sort_order, params.extended
            ))
        }
        Mode::VisualColumn => {
            // sort rows by selected columns, using the first row of the selection as the key
            // Default to cursor row if selection doesn't define a specific row range
            let sort_key_column_index = c1; // c1 is the start column of the selection

            // use all rows, since we are sorting by column
            r1 = 0; // Start from the first row
            r2 = app.rows.saturating_sub(1); // End at the last row
            c1 = app
                .visual_start
                .unwrap_or((app.cursor_row, app.cursor_col))
                .1
                .min(app.cursor_col);
            c2 = app
                .visual_start
                .unwrap_or((app.cursor_row, app.cursor_col))
                .1
                .max(app.cursor_col);

            let mut row_indices_to_sort: Vec<usize> = (r1..=r2).collect();
            row_indices_to_sort.sort_by_cached_key(|&row_idx| {
                app.cells[row_idx][sort_key_column_index.min(app.cols.saturating_sub(1))]
                    .to_content()
                    .clone()
            });
            if params.sort_order == SortOrder::Descending {
                row_indices_to_sort.reverse();
            }
            row_indices_to_sort.sort_by(|&row_a_idx, &row_b_idx| {
                let val_a = &app.cells[row_a_idx]
                    [sort_key_column_index.min(app.cols.saturating_sub(1))]
                .to_content();
                let val_b = &app.cells[row_b_idx]
                    [sort_key_column_index.min(app.cols.saturating_sub(1))]
                .to_content();
                compare_values(val_a, val_b, params.sort_type, params.sort_order)
            });
            let mut new_cells_state = app.cells.clone();
            if params.extended {
                // Sort entire application rows based on the key column from the selection
                for (new_relative_idx, &original_row_idx) in row_indices_to_sort.iter().enumerate()
                {
                    new_cells_state[new_relative_idx] = app.cells[original_row_idx].clone();
                }
            } else {
                // Sort rows only within the column bounds (c1..=c2) of the selection
                let original_block_data: Vec<Vec<Cell>> = row_indices_to_sort
                    .iter()
                    .map(|&orig_row_idx| {
                        (c1..=c2)
                            .map(|col_idx| app.cells[orig_row_idx][col_idx].clone())
                            .collect()
                    })
                    .collect();

                for (new_relative_idx, sorted_row_data_for_block) in
                    original_block_data.iter().enumerate()
                {
                    for (col_offset, cell_data) in sorted_row_data_for_block.iter().enumerate() {
                        new_cells_state[new_relative_idx][c1 + col_offset] = cell_data.clone();
                    }
                }
            }

            app.cells = new_cells_state;
            Ok(format!(
                "Sorted rows {}-{} by column {} ({:?}, {:?}, ext={})",
                r1, r2, sort_key_column_index, params.sort_type, params.sort_order, params.extended
            ))
        }
        Mode::VisualBlock => {
            // Sort rows r1..=r2 within the block. Key is the leftmost column of the block (c1 from selection).
            let sort_key_column_index = c1; // c1 is the start column of the block
            r1 = app
                .visual_start
                .unwrap_or((app.cursor_row, app.cursor_col))
                .0
                .min(app.cursor_row);
            r2 = app
                .visual_start
                .unwrap_or((app.cursor_row, app.cursor_col))
                .0
                .max(app.cursor_row);
            c1 = app
                .visual_start
                .unwrap_or((app.cursor_row, app.cursor_col))
                .1
                .min(app.cursor_col);
            c2 = app
                .visual_start
                .unwrap_or((app.cursor_row, app.cursor_col))
                .1
                .max(app.cursor_col);

            let mut row_indices_to_sort: Vec<usize> = (r1..=r2).collect();
            row_indices_to_sort.sort_by_cached_key(|&row_idx| {
                app.cells[row_idx][sort_key_column_index.min(app.cols.saturating_sub(1))]
                    .to_content()
                    .clone()
            });
            if params.sort_order == SortOrder::Descending {
                row_indices_to_sort.reverse();
            }
            row_indices_to_sort.sort_by(|&row_a_idx, &row_b_idx| {
                let val_a = &app.cells[row_a_idx]
                    [sort_key_column_index.min(app.cols.saturating_sub(1))]
                .to_content();
                let val_b = &app.cells[row_b_idx]
                    [sort_key_column_index.min(app.cols.saturating_sub(1))]
                .to_content();
                compare_values(val_a, val_b, params.sort_type, params.sort_order)
            });

            let mut new_cells_state = app.cells.clone();
            if params.extended {
                // Sort entire application rows based on the key column from the block selection
                for (new_relative_idx, &original_row_idx) in row_indices_to_sort.iter().enumerate()
                {
                    new_cells_state[r1 + new_relative_idx] = app.cells[original_row_idx].clone();
                }
            } else {
                // Sort rows only within the column bounds (c1..=c2) of the block
                let original_block_data: Vec<Vec<Cell>> = row_indices_to_sort
                    .iter()
                    .map(|&orig_row_idx| {
                        (c1..=c2)
                            .map(|col_idx| app.cells[orig_row_idx][col_idx].clone())
                            .collect()
                    })
                    .collect();

                for (new_relative_idx, sorted_row_data_for_block) in
                    original_block_data.iter().enumerate()
                {
                    for (col_offset, cell_data) in sorted_row_data_for_block.iter().enumerate() {
                        new_cells_state[r1 + new_relative_idx][c1 + col_offset] = cell_data.clone();
                    }
                }
            }
            app.cells = new_cells_state;
            Ok(format!(
                "Sorted block (rows {}-{}) by its column {} ({:?}, {:?}, ext={})",
                r1, r2, sort_key_column_index, params.sort_type, params.sort_order, params.extended
            ))
        }
        _ => Err("Sort is only available from Visual Row, Column, or Block mode.".to_string()),
    }
}

fn compare_values(
    a_str: &str,
    b_str: &str,
    sort_type: SortType,
    sort_order: SortOrder,
) -> std::cmp::Ordering {
    let ord = match sort_type {
        SortType::Numeric => {
            let a_val = a_str.trim().parse::<f64>().unwrap_or(f64::NAN);
            let b_val = b_str.trim().parse::<f64>().unwrap_or(f64::NAN);
            // Handle NANs: treat them as less than any number, or equal if both are NAN
            if a_val.is_nan() && b_val.is_nan() {
                std::cmp::Ordering::Equal
            } else if a_val.is_nan() {
                std::cmp::Ordering::Less
            }
            // NANs come first in ascending
            else if b_val.is_nan() {
                std::cmp::Ordering::Greater
            } else {
                a_val
                    .partial_cmp(&b_val)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        SortType::String => a_str.cmp(b_str),
        SortType::Length => a_str.len().cmp(&b_str.len()),
    };

    if sort_order == SortOrder::Descending {
        ord.reverse()
    } else {
        ord
    }
}

// --- End Sort Command Structures ---

// --- Popup Menu Widget ---
#[derive(Clone, Debug)]
pub struct PopupMenu {
    title: String,
    items: Vec<String>, // Items to display and select from
    pub selected_index: usize,
    scroll_offset: usize,
    max_display_items: usize, // How many items to show at once
}

impl PopupMenu {
    pub fn draw(&self, f: &mut ratatui::Frame, area: ratatui::layout::Rect) {
        // Draw a clear background for the popup
        f.render_widget(Clear, area);

        let block = Block::default()
            .title(self.title.as_str())
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow));

        let visible_items: Vec<ListItem> = self
            .items
            .iter()
            .skip(self.scroll_offset)
            .take(self.max_display_items)
            .enumerate()
            .map(|(i, item)| {
                let idx = self.scroll_offset + i;
                let style = if idx == self.selected_index {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };
                ListItem::new(item.clone()).style(style)
            })
            .collect();

        let list = List::new(visible_items).block(block).highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

        f.render_widget(list, area);
    }
}

#[derive(Debug)]
pub enum MenuWidgetEvent {
    ItemSelected(String), // Returns the selected item's string value
    Cancelled,
    Pending, // Menu is active and awaiting further input
}

impl PopupMenu {
    pub fn new(title: String, items: Vec<String>, max_display_items: usize) -> Self {
        let menu = PopupMenu {
            title,
            items,
            selected_index: 0,
            scroll_offset: 0,
            max_display_items,
        };
        if menu.items.is_empty() {
            // Add a dummy item if empty, to prevent panic and show something
            // menu.items.push("[No suggestions]".to_string());
            // Or, the caller should ensure items is not empty, or handle it.
            // For now, let's assume items might be empty and handle in draw/key_handle.
        }
        menu
    }

    pub fn handle_key(&mut self, key_code: KeyCode) -> MenuWidgetEvent {
        if self.items.is_empty() {
            return MenuWidgetEvent::Cancelled; // No items, so cancel immediately
        }
        match key_code {
            KeyCode::Up => {
                self.selected_index = self.selected_index.saturating_sub(1);
                if self.selected_index < self.scroll_offset {
                    self.scroll_offset = self.selected_index;
                }
                MenuWidgetEvent::Pending
            }
            KeyCode::Down | KeyCode::Tab => {
                if self.selected_index + 1 < self.items.len() {
                    self.selected_index += 1;
                    if self.selected_index >= self.scroll_offset + self.max_display_items {
                        self.scroll_offset += 1;
                    }
                }
                MenuWidgetEvent::Pending
            }
            KeyCode::Enter => {
                if !self.items.is_empty() {
                    MenuWidgetEvent::ItemSelected(self.items[self.selected_index].clone())
                } else {
                    MenuWidgetEvent::Cancelled
                }
            }
            KeyCode::Esc => MenuWidgetEvent::Cancelled,
            _ => MenuWidgetEvent::Pending,
        }
    }
}

// make a small window widget to show text, should have j/k to scroll through text and q to close
// on the borders, should have a title

#[derive(Debug)]
pub struct HelpPopup {
    title: String,
    content: String,
    scroll_position: usize, // Current scroll position
}

impl HelpPopup {
    pub fn new(title: &str, content: &str) -> Self {
        HelpPopup {
            title: title.to_string(),
            content: content.to_string(),
            scroll_position: 0, // Initialize scroll_position
        }
    }

    // scroll up
    pub fn scroll_up(&mut self) {
        if self.scroll_position > 0 {
            self.scroll_position -= 1;
        }
    }

    // scroll down
    pub fn scroll_down(&mut self) {
        let lines: Vec<&str> = self.content.lines().collect();
        if self.scroll_position < lines.len().saturating_sub(1) {
            self.scroll_position += 1;
        }
    }

    pub fn draw(&self, f: &mut ratatui::Frame, area: Rect) {
        // Calculate how many lines are in the content
        let content_lines: Vec<&str> = self.content.lines().collect();
        let total_lines = content_lines.len();

        // Use scroll_position from the struct
        let scroll_position = self.scroll_position;

        // Calculate max visible lines (accounting for borders and title)
        let max_visible_lines = area.height.saturating_sub(2) as usize;

        // Show scroll indicators in title if needed
        let title = if total_lines > max_visible_lines {
            format!("{} [j/k to scroll]", self.title)
        } else {
            self.title.clone()
        };

        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow));

        // Create a scrollable view of the content
        let content_view = content_lines
            .iter()
            .skip(scroll_position)
            .take(max_visible_lines).copied() // Convert &str to String or use as is if Paragraph can handle Vec<Line>
            .collect::<Vec<&str>>() // Collect as Vec<&str>
            .join("\n"); // Join into a single String

        let paragraph = Paragraph::new(content_view)
            .block(block)
            .wrap(ratatui::widgets::Wrap { trim: true });

        f.render_widget(Clear, area); // Clear background first
        f.render_widget(paragraph, area);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let args: Vec<String> = env::args().collect();
    let initial_file_to_load = if args.len() > 1 {
        Some(args[1].clone())
    } else {
        None
    };

    let res = run_app(&mut terminal, initial_file_to_load);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err);
    }

    Ok(())
}

pub enum CommandStatus {
    Success(String),
    Exit,
    Error(String),
}

fn handle_command(app: &mut App, command_string: &str) -> CommandStatus {
    // Split command_string into parts, supporting quoted arguments
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let chars = command_string.chars().peekable();

    for c in chars {
        match c {
            '"' => {
                in_quotes = !in_quotes;
                // Don't include the quote in the argument
            }
            ' ' | '\t' if !in_quotes => {
                if !current.is_empty() {
                    parts.push(current.clone());
                    current.clear();
                }
                // Skip the space
            }
            _ => {
                current.push(c);
            }
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }

    let command_name_opt = parts.first().map(|s| s.as_str());
    if command_name_opt.is_none() {
        app.command_msg = "No command entered.".to_string();
        return CommandStatus::Success(app.command_msg.clone());
    }
    let args: Vec<&str> = parts.iter().skip(1).map(|s| s.as_str()).collect();

    // Handle sort command separately as it depends heavily on visual mode context
    if command_name_opt == Some("sort") {
        // Allow with or without leading colon for sort
        if app.visual_mode_for_command.is_none() {
            app.command_msg =
                "Sort command requires an active visual selection (V, Ctrl+V, or v).".to_string();
            return CommandStatus::Success(app.command_msg.clone());
        }

        match parse_sort_flags(&args) {
            Ok(sort_params) => {
                // selected_range() already considers the visual mode for its interpretation.
                // However, the sort logic needs the original visual mode to determine primary sort axis.
                if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                    let original_visual_mode = app
                        .visual_mode_for_command
                        .clone()
                        .unwrap_or(app.mode.clone()); // Prefer stored, fallback to current

                    match execute_sort(app, original_visual_mode, r1, c1, r2, c2, sort_params) {
                        Ok(msg) => app.command_msg = msg,
                        Err(e) => app.command_msg = format!("Sort error: {}", e),
                    }
                } else {
                    // This case should ideally not be hit if visual_start is Some, but as a fallback.
                    app.command_msg = "Sort failed: No valid selection range found.".to_string();
                }
                // Clean up after sort attempt
                app.mode = Mode::Normal;
                app.visual_start = None;

                app.input.clear();
                return CommandStatus::Success(app.command_msg.clone());
            }
            Err(e) => {
                app.command_msg = format!("Sort parse error: {}", e);
                // Don't necessarily change mode if parsing fails, let user correct.
                return CommandStatus::Success(app.command_msg.clone());
            }
        }
    }

    if let Some(command_name) = command_name_opt {
        if let Some(sel_cmd) = get_selection_command(command_name) {
            if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                if sel_cmd.modifies_data() {
                    app.snapshot(); // Call snapshot before executing if data is modified
                }
                match sel_cmd.execute(app, r1, c1, r2, c2) {
                    Ok(msg) => app.command_msg = msg,
                    Err(e) => app.command_msg = format!("Error: {}", e),
                }
            } else {
                app.command_msg = format!("Command '{}' requires a selection.", sel_cmd.name());
            }
            return CommandStatus::Success(app.command_msg.clone());
        }
    }

    match command_name_opt {
        Some("q") => {
            app.command_msg = "Exiting...".to_string();
            return CommandStatus::Exit;
        }
        Some("w") | Some("write") => {
            let save_path = if let Some(path) = args.first() {
                Some(path.to_string())
            } else { app.filepath.as_ref().map(|filepath| filepath.to_string_lossy().to_string()) };
            if let Some(path) = save_path {
                match save_csv(app, &path) {
                    Ok(_) => app.command_msg = format!("Saved to {}", path),
                    Err(e) => app.command_msg = format!("Error saving: {}", e),
                }
            } else {
                app.command_msg = "Usage: w <path> (or load a file first)".to_string();
            }
        }
        Some("files") => {
            // List files in the current directory and show in popup
            let current_dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            match fs::read_dir(&current_dir) {
                Ok(entries) => {
                    let mut file_list = String::new();
                    for entry in entries.flatten() {
                        if let Ok(file_name) = entry.file_name().into_string() {
                            file_list.push_str(&format!("{}\n", file_name));
                        }
                    }
                    let help_popup =
                        HelpPopup::new(&format!("Files in {}", current_dir.display()), &file_list);
                    app.help_popup = Some(help_popup);
                    app.command_msg = format!(
                        "Showing files in {}. Press Esc, q, or h to close.",
                        current_dir.display()
                    );
                }
                Err(e) => app.command_msg = format!("Error reading directory: {}", e),
            }
        }
        Some("wq") => {
            let save_path = if let Some(path) = args.first() {
                Some(path.to_string())
            } else { app.filepath.as_ref().map(|filepath| filepath.to_string_lossy().to_string()) };
            if let Some(path) = save_path {
                match save_csv(app, &path) {
                    Ok(_) => app.command_msg = format!("Saved to {}", path),
                    Err(e) => app.command_msg = format!("Error saving: {}", e),
                }
            } else {
                app.command_msg = "Usage: wq <path> (or load a file first)".to_string();
            }
            app.command_msg += " Exiting...";
            return CommandStatus::Exit;
        }
        // num -> turns all cells to numbers, if cell is string, it will try to parse it as a number, if formula, it will keep it as text
        Some("num") | Some("number") => {
            app.snapshot();
            for row in &mut app.cells {
                for cell in row {
                    if let CellValue::Text(ref text) = cell.value {
                        if let Ok(num) = text.trim().parse::<f64>() {
                            cell.value = CellValue::Number(num);
                        } else {
                            cell.value = CellValue::Text(text.clone()); // Keep original text if parse fails
                        }
                    }
                }
            }
            app.command_msg = "Converted all cells to numbers where possible".to_string();
        }
        // goto <row> [: <col>]
        Some("goto") => {
            if let Some(row_str) = args.first() {
                if let Ok(row) = row_str.parse::<usize>() {
                    let col = if args.len() > 1 {
                        args[1].parse::<usize>().unwrap_or(0)
                    } else {
                        0
                    };
                    if row < app.rows && col < app.cols {
                        app.cursor_row = row;
                        app.cursor_col = col;
                        app.scroll_row = row.saturating_sub(app.view_rows / 2);
                        app.scroll_col = col.saturating_sub(app.view_cols / 2);
                        app.command_msg = format!("Moved cursor to ({}, {})", row, col);
                    } else {
                        app.command_msg = "Row or column out of bounds".to_string();
                    }
                } else {
                    app.command_msg = "Invalid row number".to_string();
                }
            } else {
                app.command_msg = "Usage: goto <row> [: <col>]".to_string();
            }
        }
        // r, jumps row
        Some("r") | Some("row") => {
            if let Some(row_str) = args.first() {
                if let Ok(row) = row_str.parse::<usize>() {
                    if row < app.rows {
                        app.cursor_row = row;
                        app.scroll_row = row.saturating_sub(app.view_rows / 2);
                        app.command_msg = format!("Moved cursor to row {}", row);
                    } else {
                        app.command_msg = "Row out of bounds".to_string();
                    }
                } else {
                    app.command_msg = "Invalid row number".to_string();
                }
            } else {
                app.command_msg = "Usage: r <row>".to_string();
            }
        }
        // c, jumps column
        Some("c") | Some("col") => {
            if let Some(col_str) = args.first() {
                if let Ok(col) = col_str.parse::<usize>() {
                    if col < app.cols {
                        app.cursor_col = col;
                        app.scroll_col = col.saturating_sub(app.view_cols / 2);
                        app.command_msg = format!("Moved cursor to column {}", col);
                    } else {
                        app.command_msg = "Column out of bounds".to_string();
                    }
                } else {
                    app.command_msg = "Invalid column number".to_string();
                }
            } else {
                app.command_msg = "Usage: c <col>".to_string();
            }
        }
        // fill, uses clipboard content to fill cells that are selected, otherwise use first arg as content
        Some("fill") => {
            if let Some(content_arg) = args.first() {
            // Argument provided, use it to fill
            if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                app.snapshot(); // Save current state before filling
                for r in r1..=r2 {
                for c in c1..=c2 {
                    if r < app.rows && c < app.cols {
                    // Try to parse as number, else store as text
                    if let Ok(num) = content_arg.parse::<f64>() {
                        app.cells[r][c].value = CellValue::Number(num);
                    } else {
                        app.cells[r][c].value = CellValue::Text(content_arg.to_string());
                    }
                    }
                }
                }
                app.command_msg = format!(
                "Filled cells ({}, {}) to ({}, {}) with '{}'",
                r1, c1, r2, c2, content_arg
                );
            } else {
                app.command_msg = "No selection to fill".to_string();
            }
            } else if !app.clipboard.is_empty() {
            // No argument, but clipboard has content, use clipboard
            // Assuming clipboard contains a single cell's content for fill for simplicity
            let clipboard_content_to_fill = app.clipboard[0][0].to_content();
            if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                app.snapshot(); // Save current state before filling
                for r in r1..=r2 {
                for c in c1..=c2 {
                    if r < app.rows && c < app.cols {
                    // Try to parse as number, else store as text
                    if let Ok(num) = clipboard_content_to_fill.parse::<f64>() {
                        app.cells[r][c].value = CellValue::Number(num);
                    } else {
                        app.cells[r][c].value = CellValue::Text(clipboard_content_to_fill.clone());
                    }
                    }
                }
                }
                app.command_msg = format!(
                "Filled cells ({}, {}) to ({}, {}) with clipboard content",
                r1, c1, r2, c2
                );
            } else {
                app.command_msg = "No selection to fill".to_string();
            }
            } else {
            // No argument and clipboard is empty
            app.command_msg = "Usage: fill <content> or copy to clipboard first".to_string();
            }
        }
        // clear, clears all cells
        Some("clear") => {
            app.snapshot();
            app.cells = vec![
                vec![
                    Cell {
                        value: CellValue::Text("".to_string())
                    };
                    app.cols
                ];
                app.rows
            ];
            app.command_msg = "Cleared all cells".to_string();
        }
        Some("undo") => {
            app.undo();
            app.command_msg = "Undone last action".to_string();
        }
        Some("redo") => {
            app.redo();
            app.command_msg = "Redone last action".to_string();
        }
        // new (r | c | rc) [number], creates new row(s) or column(s) at cursor or selection
        Some("new") => {
            if args.is_empty() {
            app.command_msg = "Usage: new <r|c|rc> [number]".to_string();
            } else {
            let count = if args.len() > 1 {
                args[1].parse::<usize>().unwrap_or(1).max(1)
            } else {
                1
            };
            match args[0] {
                "r" | "row" => {
                app.snapshot();
                let insert_at = if let Some(((r1, _), (r2, _))) = app.selected_range() {
                    r1.min(r2)
                } else {
                    app.cursor_row
                };
                for _ in 0..count {
                    app.insert_row(insert_at);
                }
                app.command_msg = format!("Inserted {} new row(s) at {}", count, insert_at);
                }
                "c" | "col" => {
                app.snapshot();
                let insert_at = if let Some(((_, c1), (_, c2))) = app.selected_range() {
                    c1.min(c2)
                } else {
                    app.cursor_col
                };
                for _ in 0..count {
                    app.insert_col(insert_at);
                }
                app.command_msg = format!("Inserted {} new column(s) at {}", count, insert_at);
                }
                "rc" | "rowcol" => {
                app.snapshot();
                let insert_row_at = if let Some(((r1, _), (r2, _))) = app.selected_range() {
                    r1.min(r2)
                } else {
                    app.cursor_row
                };
                let insert_col_at = if let Some(((_, c1), (_, c2))) = app.selected_range() {
                    c1.min(c2)
                } else {
                    app.cursor_col
                };
                for _ in 0..count {
                    app.insert_row(insert_row_at);
                    app.insert_col(insert_col_at);
                }
                app.command_msg = format!(
                    "Inserted {} new row(s) at {} and column(s) at {}",
                    count, insert_row_at, insert_col_at
                );
                }
                _ => {
                app.command_msg = "Invalid argument for new command".to_string();
                }
            }
            }
        }
        // delete (r | c | rc), deletes row(s) or column(s) at cursor or selection
        Some("delete") => {
            if args.is_empty() {
            app.command_msg = "Usage: delete <r|c|rc>".to_string();
            } else {
            match args[0] {
                "r" | "row" => {
                let (start, end) = if let Some(((r1, _), (r2, _))) = app.selected_range() {
                    (r1.min(r2), r1.max(r2))
                } else {
                    (app.cursor_row, app.cursor_row)
                };
                let num_to_delete = end - start + 1;
                if app.rows > num_to_delete {
                    app.snapshot();
                    for _ in start..=end {
                    app.delete_row(start);
                    }
                    app.command_msg = format!("Deleted row(s) {} to {}", start, end);
                } else {
                    app.command_msg = "Cannot delete all rows".to_string();
                }
                }
                "c" | "col" => {
                let (start, end) = if let Some(((_, c1), (_, c2))) = app.selected_range() {
                    (c1.min(c2), c1.max(c2))
                } else {
                    (app.cursor_col, app.cursor_col)
                };
                let num_to_delete = end - start + 1;
                if app.cols > num_to_delete {
                    app.snapshot();
                    for _ in start..=end {
                    app.delete_col(start);
                    }
                    app.command_msg = format!("Deleted column(s) {} to {}", start, end);
                } else {
                    app.command_msg = "Cannot delete all columns".to_string();
                }
                }
                "rc" | "rowcol" => {
                let (row_start, row_end) = if let Some(((r1, _), (r2, _))) = app.selected_range() {
                    (r1.min(r2), r1.max(r2))
                } else {
                    (app.cursor_row, app.cursor_row)
                };
                let (col_start, col_end) = if let Some(((_, c1), (_, c2))) = app.selected_range() {
                    (c1.min(c2), c1.max(c2))
                } else {
                    (app.cursor_col, app.cursor_col)
                };
                let num_rows = row_end - row_start + 1;
                let num_cols = col_end - col_start + 1;
                if app.rows > num_rows && app.cols > num_cols {
                    app.snapshot();
                    for _ in row_start..=row_end {
                    app.delete_row(row_start);
                    }
                    for _ in col_start..=col_end {
                    app.delete_col(col_start);
                    }
                    app.command_msg = format!(
                    "Deleted row(s) {} to {} and column(s) {} to {}",
                    row_start, row_end, col_start, col_end
                    );
                } else {
                    app.command_msg = "Cannot delete all rows or columns".to_string();
                }
                }
                _ => {
                app.command_msg = "Invalid argument for delete command".to_string();
                }
            }
            }
        }

        Some("save") => {
            if let Some(path) = args.first() {
                match save_csv(app, path) {
                    Ok(_) => app.command_msg = format!("Saved to {}", path),
                    Err(e) => app.command_msg = format!("Error saving: {}", e),
                }
            } else {
                app.command_msg = "Usage: save <path>".to_string();
            }
        }
        Some("load") => {
            if let Some(path) = args.first() {
                match load_csv(app, path) {
                    Ok(_) => {
                        app.command_msg = format!("Loaded from {}", path);
                        app.filepath = Some(PathBuf::from(path));
                    }
                    Err(e) => app.command_msg = format!("Error loading: {}", e),
                }
            } else {
                app.command_msg = "Usage: load <path>".to_string();
            }
        }
        Some("help") => {
            if app.help_popup.is_some() {
                app.help_text = None;
                app.command_msg = "Help closed.".to_string();
            } else {
                app.help_popup = Some(HelpPopup::new("Help", HELP_MESSAGE));
                app.help_text = Some(HELP_MESSAGE.to_string());
                app.command_msg = "Showing help. Press Esc, q, or h to close.".to_string();
            }
        }
        Some(cmd) => {
            app.command_msg = format!("Unknown command: {}", cmd);
        }
        None => {
            app.command_msg = "No command entered".to_string();
        }
    }
    CommandStatus::Success(app.command_msg.clone())
}

fn save_csv(app: &App, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut wtr = csv::Writer::from_writer(file);

    let mut actual_rows = 0;
    let mut actual_cols = 0;

    // Determine the actual extent of data
    for r in 0..app.rows {
        let mut row_has_data = false;
        for c in 0..app.cols {
            if !app.cells[r][c].to_content().is_empty() {
                row_has_data = true;
                actual_cols = actual_cols.max(c + 1);
            }
        }
        if row_has_data {
            actual_rows = r + 1;
        }
    }

    // Collect all cell contents as Strings first to manage lifetimes
    let mut all_cell_data: Vec<Vec<String>> = Vec::with_capacity(actual_rows);
    for r_idx in 0..actual_rows {
        let row_data: Vec<String> = (0..actual_cols)
            .map(|c_idx| {
                if r_idx < app.rows && c_idx < app.cols {
                    app.cells[r_idx][c_idx].to_content()
                } else {
                    String::new()
                }
            })
            .collect();
        all_cell_data.push(row_data);
    }

    for r_idx in 0..actual_rows {
        // Now create a record of string slices from the owned Strings
        let record: Vec<&str> = (0..actual_cols)
            .map(|c_idx| all_cell_data[r_idx][c_idx].as_str())
            .collect();
        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    Ok(())
}

fn load_csv(app: &mut App, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    // Explicitly state that the CSV may not have headers, or handle them if they exist
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    app.snapshot(); // Save current state before loading

    let mut records_data: Vec<Vec<String>> = Vec::new();
    let mut csv_num_rows = 0;
    let mut csv_max_cols = 0;

    for result in rdr.records() {
        let record = result?;
        csv_num_rows += 1;
        csv_max_cols = csv_max_cols.max(record.len());
        records_data.push(record.iter().map(|field| field.to_string()).collect());
    }

    if csv_num_rows == 0 {
        // Handle empty CSV file
        app.cells = vec![
            vec![
                Cell {
                    value: CellValue::Text("".to_string())
                };
                1
            ];
            1
        ]; // Default to 1x1
        app.rows = 1;
        app.cols = 1;
        app.command_msg = "Loaded empty CSV. Sheet reset to 1x1.".to_string();
    } else {
        app.rows = csv_num_rows;
        app.cols = csv_max_cols;
        app.cells = vec![
            vec![
                Cell {
                    value: CellValue::Text("".to_string())
                };
                app.cols
            ];
            app.rows
        ];

        for (r, row_data) in records_data.iter().enumerate() {
            for (c, field_content) in row_data.iter().enumerate() {
                if r < app.rows && c < app.cols {
                    // Try to parse as number, else store as text
                    if let Ok(num) = field_content.trim().parse::<f64>() {
                        app.cells[r][c].value = CellValue::Number(num);
                    } else {
                        app.cells[r][c].value = CellValue::Text(field_content.clone());
                    }
                }
            }
        }
        app.command_msg = format!("Loaded {}x{} cells from {}", app.rows, app.cols, path);
    }

    // Reset cursor and scroll to default positions
    app.cursor_row = 0;
    app.cursor_col = 0;
    app.scroll_row = 0;
    app.scroll_col = 0;

    Ok(())
}

// Sorting mode and order enums
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum SortMode {
    Number,
    String,
    Length,
}

// Remove this duplicate SortOrder definition to avoid conflict
// #[derive(Debug, PartialEq, Eq, Clone, Copy)]
// enum SortOrder {
//     Ascending,
//     Descending,
// }

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    initial_file_to_load: Option<String>,
) -> io::Result<()> {
    // Initialize the application with default size
    let mut app = App::new(100, 26);

    // The following methods are already implemented on App:
    // - app.snapshot(): Take a snapshot of the current state for undo/redo.
    // - app.undo(): Undo the last action.
    // - app.redo(): Redo the last undone action.

    // Example usage (not needed here, just for reference):
    // app.snapshot();
    // app.undo();
    // app.redo();

    if let Some(path) = initial_file_to_load {
        match load_csv(&mut app, &path) {
            Ok(_) => app.command_msg = format!("Loaded from {}", path),
            Err(e) => app.command_msg = format!("Error loading initial file {}: {}", path, e),
        }
    }

    loop {
        terminal.draw(|f| {
            let size = f.size();

            if app.help_popup.is_some() {
                // If help popup is active, draw it
                if let Some(help_popup) = &app.help_popup {
                    help_popup.draw(f, size);
                }
                return; // Skip the rest of the drawing if help popup is active
            }

            // Create cell_map for formula evaluation, once per draw call
            let mut cell_map_for_eval: HashMap<(usize, usize), &CellValue> = HashMap::new();
            if app.rows > 0 && app.cols > 0 {
                for (r_idx, row_vec) in app.cells.iter().enumerate() {
                    for (c_idx, cell_obj) in row_vec.iter().enumerate() {
                        cell_map_for_eval.insert((r_idx, c_idx), &cell_obj.value);
                    }
                }
            }

            // Main application layout (spreadsheet, inspector, command bar)
            // Adjust view_rows for the new inspector bar and command bar
            app.view_rows = (size.height as usize).saturating_sub(3);

            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(
                    [
                        Constraint::Min(0),    // Spreadsheet area
                        Constraint::Length(1), // Inspector Bar
                        Constraint::Length(1), // Command Bar / Footer
                    ]
                    .as_ref(),
                )
                .split(size);

            let main_area = layout[0];
            let inspector_area = layout[1];
            let footer_area = layout[2];

            // Calculate number_width (for row numbers) first
            let number_width = app.rows.to_string().len();
            let available_width_for_cells = (main_area.width as usize).saturating_sub(number_width);
            let cell_column_display_width = app.cwidth + 2;
            if cell_column_display_width > 0 {
                app.view_cols = available_width_for_cells / cell_column_display_width;
            } else {
                app.view_cols = 0;
            }
            // Ensure view_rows is also updated based on the main_area height for spreadsheet
            app.view_rows = (main_area.height as usize).saturating_sub(3); // -1 for header row

            let mut lines = vec![];
            // Helper function to convert 0-indexed column number to Excel-style string (A, B, ..., Z, AA, etc.)
            // This function should be defined in an accessible scope, e.g., within main.rs.
            #[allow(dead_code)] // Remove if used, or ensure it's used.
            fn to_excel_col(idx: usize) -> String {
                let mut temp_idx = idx;
                let mut name = String::new();
                loop {
                    name.insert(0, (b'A' + (temp_idx % 26) as u8) as char);
                    if temp_idx < 26 {
                        break;
                    }
                    temp_idx = temp_idx / 26 - 1;
                }
                name
            }

            // Create header spans
            let mut header_spans: Vec<Span> = Vec::new();
            // Add an empty span for the row number column, matching number_width
            header_spans.push(Span::styled(
                format!("{:width$}", "", width = number_width),
                Style::default(), // This corner piece doesn't get highlighted by column selection
            ));

            for c in app.scroll_col..(app.scroll_col + app.view_cols).min(app.cols) {
                let col_name = to_excel_col(c);
                // Center the column name within app.cwidth, add 1 space padding on each side
                let header_text = format!(" {:^width$} ", col_name, width = app.cwidth);

                let mut is_col_highlighted = false;

                // Check if the current column `c` is the cursor's column
                if app.cursor_col == c {
                    is_col_highlighted = true;
                }

                // Check if the current column `c` is within the selected range's columns
                if !is_col_highlighted { // Only check if not already highlighted by cursor
                    if let Some(((_sel_r1, sel_c1), (_sel_r2, sel_c2))) = app.selected_range() {
                        if c >= sel_c1 && c <= sel_c2 {
                            is_col_highlighted = true;
                        }
                    }
                }

                let header_style = if is_col_highlighted {
                    Style::default().add_modifier(Modifier::REVERSED) // Highlight style
                } else {
                    Style::default() // Default style
                };

                header_spans.push(Span::styled(header_text, header_style));
            }
            lines.push(Line::from(header_spans));

            for r in app.scroll_row..(app.scroll_row + app.view_rows).min(app.rows) {
                let mut row_spans: Vec<Span> = Vec::new();
                // Highlight if cursor is on this row
                let mut is_row_highlighted = false;
                if r == app.cursor_row {
                    is_row_highlighted = true;
                }

                // Highlight if this row is part of a visual selection's vertical span
                if !is_row_highlighted {
                    if let Some(((sel_r1, _), (sel_r2, _))) = app.selected_range() {
                        if r >= sel_r1 && r <= sel_r2 {
                            is_row_highlighted = true;
                        }
                    }
                }

                let row_number_text = format!("{:width$}", r, width = number_width);
                let row_number_style = if is_row_highlighted {
                    Style::default().add_modifier(Modifier::REVERSED)
                } else {
                    Style::default()
                };
                row_spans.push(Span::styled(row_number_text, row_number_style));

                for c in app.scroll_col..(app.scroll_col + app.view_cols).min(app.cols) {
                    let (mut display_text, mut base_style, is_formula) = if r < app.rows && c < app.cols {
                        let cell = &app.cells[r][c];
                        match &cell.value {
                            CellValue::Formula(expr_str) => {
                                let mut formula_display = "ƒ".to_string();
                                if let Some(expr) = parse_expr(expr_str) {
                                    let mut visited = HashSet::new();
                                    let eval_result = eval_expr(&expr, &cell_map_for_eval, &mut visited);
                                    if eval_result.is_nan() {
                                        formula_display.push_str("#DIV/0!");
                                        (formula_display, Style::default().fg(Color::Red).add_modifier(Modifier::BOLD), true)
                                    } else if eval_result.is_infinite() {
                                        formula_display.push_str("#INF!");
                                        (formula_display, Style::default().fg(Color::Red).add_modifier(Modifier::BOLD), true)
                                    } else {
                                        formula_display.push_str(&format!("{:.2}", eval_result));
                                        (formula_display, Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC), true)
                                    }
                                } else {
                                    formula_display.push_str("#P_ERR");
                                    (formula_display, Style::default().fg(Color::Red).add_modifier(Modifier::BOLD), true)
                                }
                            }
                            CellValue::Number(n) => (n.to_string(), Style::default().fg(Color::Cyan), false),
                            CellValue::Text(s) => (s.clone(), Style::default(), false),
                        }
                    } else {
                        ("".to_string(), Style::default(), false)
                    };

                    // Truncate and center the display_text to app.cwidth
                    let truncated = display_text.chars().take(app.cwidth).collect::<String>();
                    let centered = format!("{:^width$}", truncated, width = app.cwidth);
                    display_text = centered;

                    // Determine if the cell is the cursor or part of a selection
                    let is_cursor = r == app.cursor_row && c == app.cursor_col;
                    let is_selected = if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                        r >= r1 && r <= r2 && c >= c1 && c <= c2
                    } else {
                        false
                    };

                    let mut final_style = base_style;
                    if is_formula {
                        final_style = final_style.fg(Color::Yellow);
                    }
                    if is_cursor {
                        final_style = final_style.add_modifier(Modifier::REVERSED);
                    } else if is_selected {
                        final_style = final_style.bg(Color::DarkGray);
                    }

                    // Always pad to fixed width (cwidth+2 for borders/padding)
                    let cell_text = if is_cursor {
                        // Show brackets for cursor cell, but keep width fixed
                        let inner = display_text.trim();
                        let pad = app.cwidth.saturating_sub(inner.len());
                        let left = pad / 2;
                        let right = pad - left;
                        format!(
                            "[{}{}{}]",
                            " ".repeat(left),
                            inner,
                            " ".repeat(right)
                        )
                        .chars()
                        .take(app.cwidth + 2)
                        .collect::<String>()
                    } else {
                        // Normal/selected cell: pad with spaces to cwidth+2
                        format!(" {} ", display_text)
                            .chars()
                            .take(app.cwidth + 2)
                            .collect::<String>()
                    };

                    row_spans.push(Span::styled(cell_text, final_style));
                }
                lines.push(Line::from(row_spans));
            }

            let title = if let Some(path) = &app.filepath {
                format!("Spreadsheet - {}", path.file_name().unwrap_or_default().to_string_lossy())
            } else {
                "Spreadsheet".to_string()
            };
            let sheet_block = Block::default().title(title).borders(Borders::ALL);
            f.render_widget(Paragraph::new(lines).block(sheet_block), main_area);


            // Inspector Bar
            let (current_cell_content, inspector_formula_info) = if app.rows > 0 && app.cols > 0 {
                let cell = &app.cells[app.cursor_row.min(app.rows - 1)][app.cursor_col.min(app.cols - 1)];
                match &cell.value {
                    CellValue::Formula(expr_str) => {
                        // Use native evaluator
                        let mut cell_map: HashMap<(usize, usize), &CellValue> = HashMap::new();
                        for (r_idx, row_vec) in app.cells.iter().enumerate() {
                            for (c_idx, cell_val) in row_vec.iter().enumerate() {
                                cell_map.insert((r_idx, c_idx), &cell_val.value);
                            }
                        }
                        if let Some(expr) = parse_expr(expr_str) {
                            let mut visited = std::collections::HashSet::new();
                            let eval_result = eval_expr(&expr, &cell_map, &mut visited);
                            let value_str = format!("{:.4}", eval_result);
                            (format!("={}", expr_str), Some(format!("= {} → {}", expr_str, value_str)))
                        } else {
                            (format!("={}", expr_str), Some("#PARSE_ERR".to_string()))
                        }
                    }
                    _ => (cell.to_content(), None),
                }
            } else {
                ("N/A".to_string(), None)
            };
            let inspector_text = if let Some(formula_info) = inspector_formula_info {
                format!(
                    "Cell ({}, {}): {} | {} | Size: {}Rx{}C",
                    app.cursor_row, app.cursor_col, current_cell_content, formula_info, app.rows, app.cols,
                )
            } else {
                format!(
                    "Cell ({}, {}): \"{}\" | Size: {}Rx{}C",
                    app.cursor_row, app.cursor_col, current_cell_content, app.rows, app.cols,
                )
            };
            let inspector_paragraph = Paragraph::new(inspector_text);
            f.render_widget(inspector_paragraph, inspector_area);

            // Footer / Command Bar
            let footer_text = match app.mode {
                Mode::Command => {
                    // Show the input and set the terminal cursor position
                    let cursor_pos = app.input.len();
                    let prompt_len = 1; // ":" prompt
                    let x = footer_area.x + prompt_len as u16 + cursor_pos as u16;
                    let y = footer_area.y;
                    f.set_cursor(x, y);
                    format!(":{}", app.input)
                }
                Mode::Insert => {
                    // Show the input and set the terminal cursor position
                    let cursor_pos = app.insert_mode_cursor.min(app.input.len());
                    let prompt_len = "INSERT ".len();
                    let x = footer_area.x + prompt_len as u16 + cursor_pos as u16;
                    let y = footer_area.y;
                    f.set_cursor(x, y);
                    format!("INSERT {}", app.input)
                }
                Mode::Normal => format!("-- NORMAL --  {}", app.command_msg),
                Mode::VisualRow => "-- VISUAL LINE --".into(),
                Mode::VisualBlock => "-- VISUAL BLOCK --".into(),
                Mode::VisualColumn => "-- VISUAL COLUMN --".into(),
            };
            f.render_widget(Paragraph::new(footer_text), footer_area);

            // Draw popup menu if active (drawn on top of everything else in its area)
            if let Some(menu) = &app.popup_menu {
                // Position the popup menu, e.g., above the command bar or centered
                // For now, a simple centered rect, or one near the command input
                let menu_height = (menu.items.len().min(menu.max_display_items) + 2) as u16; // +2 for borders
                let menu_width = (menu
                    .title
                    .len()
                    .max(menu.items.iter().map(|s| s.len()).max().unwrap_or(10))
                    + 4) as u16; // +4 for padding/borders

                // Use the footer_area from the layout to position the popup menu above the command bar
                let popup_area = {
                    let x = footer_area.x + 2;
                    let y = footer_area.y.saturating_sub(menu_height);
                    let width = menu_width.min(size.width.saturating_sub(4));
                    let height = menu_height.min(size.height.saturating_sub(1));
                    Rect {
                        x,
                        y,
                        width,
                        height,
                    }
                };
                menu.draw(f, popup_area);
            }
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // look for ctrl-c
                if key.code == KeyCode::Char('q') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    app.command_msg = "Exiting...".to_string();
                    return Ok(()); // Exit the application
                }
                if app.help_popup.is_some() {
                    match key.code {
                        KeyCode::Esc
                        | KeyCode::Char('q')
                        | KeyCode::Char('h')
                        | KeyCode::Char('H') => {
                            app.help_popup = None;
                            app.help_text = None;
                            app.command_msg = "Help closed.".to_string();
                        }
                        KeyCode::Char('j') => {
                            // Scroll down in help
                            if let Some(help_popup) = &mut app.help_popup {
                                help_popup.scroll_down();
                            }
                        }
                        KeyCode::Char('k') => {
                            // Scroll up in help
                            if let Some(help_popup) = &mut app.help_popup {
                                help_popup.scroll_up();
                            }
                        }
                        KeyCode::Char('g') => {
                            // Go to top of help
                            if let Some(help_popup) = &mut app.help_popup {
                                help_popup.scroll_position = 0;
                            }
                        }
                        _ => {} // Other keys do nothing when help is open
                    }
                } else if let Some(menu) = app.popup_menu.as_mut() {
                    // Input is directed to the popup menu
                    match menu.handle_key(key.code) {
                        MenuWidgetEvent::ItemSelected(selected_item_value) => {
                            let mut new_input = app.tab_completion_command_prefix.clone();
                            new_input.push_str(&app.tab_completion_path_prefix);
                            new_input.push_str(&selected_item_value);

                            // Check if the completed item is a directory
                            let completed_path = PathBuf::from(format!(
                                "{}{}",
                                app.tab_completion_path_prefix, selected_item_value
                            ));
                            if completed_path.is_dir() {
                                new_input.push('/');
                            }
                            app.input = new_input;
                            app.popup_menu = None;
                        }
                        MenuWidgetEvent::Cancelled => {
                            app.popup_menu = None;
                            // Optionally restore app.input to what it was before tab was pressed,
                            // or just leave it as is for the user to continue editing.
                        }
                        MenuWidgetEvent::Pending => {
                            // Menu is still active, do nothing more here
                        }
                    }
                } else {
                    match app.mode {
                        Mode::Normal => match key.code {
                            // o/O should open new row below/above
                            KeyCode::Char('o') if key.modifiers.is_empty() => {
                                app.snapshot();
                                app.insert_row(app.cursor_row + 1);
                                app.cursor_row += 1;
                                app.command_msg = format!(
                                    "New row inserted below row {}",
                                    app.cursor_row
                                );
                            }
                            KeyCode::Char('O') if key.modifiers.contains(KeyModifiers::SHIFT) => {
                                app.snapshot();
                                app.insert_row(app.cursor_row);
                                app.command_msg = format!(
                                    "New row inserted above row {}",
                                    app.cursor_row
                                );
                            }
                            // a/A should open new column right/left    
                            KeyCode::Char('a') if key.modifiers.is_empty() => {
                                app.snapshot();
                                app.insert_col(app.cursor_col + 1);
                                app.cursor_col += 1;
                                app.command_msg = format!(
                                    "New column inserted right of column {}",
                                    app.cursor_col
                                );
                            }
                            KeyCode::Char('A') if key.modifiers.contains(KeyModifiers::SHIFT) => {
                                app.snapshot();
                                app.insert_col(app.cursor_col);
                                app.command_msg = format!(
                                    "New column inserted left of column {}",
                                    app.cursor_col
                                );
                            }

                            // D should delete row
                            KeyCode::Char('D') if key.modifiers.contains(KeyModifiers::SHIFT) => {
                                let row_to_delete = app.cursor_row;
                                if app.rows > 1 {
                                    app.snapshot();
                                    app.delete_row(row_to_delete);
                                    app.command_msg = format!("Deleted row {}", row_to_delete);
                                    // Adjust cursor position if needed
                                    if app.cursor_row >= app.rows {
                                        app.cursor_row = app.rows.saturating_sub(1);
                                    }
                                } else {
                                    app.command_msg = "Cannot delete the last row.".to_string();
                                }
                            }
                            // ctrl + d should delete column
                            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                let col_to_delete = app.cursor_col;
                                if app.cols > 1 {
                                    app.snapshot();
                                    app.delete_col(col_to_delete);
                                    app.command_msg = format!("Deleted column {}", col_to_delete);
                                    // Adjust cursor position if needed
                                    if app.cursor_col >= app.cols {
                                        app.cursor_col = app.cols.saturating_sub(1);
                                    }
                                } else {
                                    app.command_msg = "Cannot delete the last column.".to_string();
                                }
                            }
                            // r for result into cell
                            KeyCode::Char('r') if key.modifiers.is_empty() => {
                                if app.result.is_some() {
                                    app.snapshot();
                                    let result = app.result.clone().unwrap();
                                    app.cells[app.cursor_row][app.cursor_col].value =
                                        CellValue::Number(result);
                                    app.command_msg = format!(
                                        "Result inserted into cell ({}, {})",
                                        app.cursor_row, app.cursor_col
                                    );
                                } else {
                                    app.command_msg = "No result to insert.".to_string();
                                }
                            }
                            // f to insert formula
                            KeyCode::Char('f') if key.modifiers.is_empty() => {
                                app.mode = Mode::Insert;
                                app.input = "=".to_string();
                                app.command_msg = "Formula mode: type your formula and press Enter".to_string();
                                // set cursor to end of input
                                app.insert_mode_cursor = app.input.len();

                            }
                            KeyCode::Char(',') if key.modifiers.is_empty() => {
                                // restore selection range
                                if let Some((start_coords, end_coords)) = app.last_selection_range {
                                    app.mode = app
                                        .visual_mode_for_command
                                        .clone()
                                        .unwrap_or(Mode::VisualBlock);
                                    app.visual_start = Some(start_coords);
                                    // cursor should be at end of last selection
                                    app.cursor_row = end_coords.0;
                                    app.cursor_col = end_coords.1;
                                } else {
                                    app.command_msg =
                                        "No previous selection to restore.".to_string();
                                }
                            }
                            // g to go to the top
                            KeyCode::Char('g') if key.modifiers.is_empty() => {
                                app.cursor_row = 0;
                            }
                            // G to go to the bottom
                            KeyCode::Char('G') if key.modifiers.contains(KeyModifiers::SHIFT) => {
                                app.cursor_row = app.rows.saturating_sub(1);
                            }
                            KeyCode::Char('h') if key.modifiers.is_empty() => {
                                app.cursor_col = app.cursor_col.saturating_sub(1);
                            }
                            KeyCode::Char('H') if key.modifiers.contains(KeyModifiers::SHIFT) => {
                                // moves cell to the left
                                if app.cursor_col > 0 {
                                    app.snapshot();
                                    let row = &mut app.cells[app.cursor_row];
                                    let (left, right) = row.split_at_mut(app.cursor_col);
                                    let left_cell = &mut left[app.cursor_col - 1];
                                    let current_cell = &mut right[0];
                                    std::mem::swap(
                                        &mut left_cell.value,
                                        &mut current_cell.value,
                                    );
                                    app.cursor_col -= 1;
                                }
                            }
                            KeyCode::Char('l') => {
                                app.cursor_col = (app.cursor_col + 1).min(app.cols - 1);
}
                            KeyCode::Char('L') if key.modifiers.contains(KeyModifiers::SHIFT) => {
                                // moves cell to the right
                                if app.cursor_col < app.cols - 1 {
                                    app.snapshot();
                                    let row = &mut app.cells[app.cursor_row];
                                    let (left, right) = row.split_at_mut(app.cursor_col + 1);
                                    let current_cell = &mut left[app.cursor_col];
                                    let right_cell = &mut right[0];
                                    std::mem::swap(
                                        &mut current_cell.value,
                                        &mut right_cell.value,
                                    );
                                    app.cursor_col += 1;
                            }
                            }
                            KeyCode::Char('0') => app.cursor_col = 0,
                            KeyCode::Char('$') => app.cursor_col = app.cols,
                            KeyCode::Char('k') => app.cursor_row = app.cursor_row.saturating_sub(1),
                            KeyCode::Char('K') => {
                                // moves cell up
                                if app.cursor_row > 0 {
                                    app.snapshot();
                                    let (above, below) = app.cells.split_at_mut(app.cursor_row);
                                    let current_cell =
                                        &mut above[app.cursor_row - 1][app.cursor_col];
                                    let below_cell = &mut below[0][app.cursor_col];
                                    std::mem::swap(
                                        &mut current_cell.value,
                                        &mut below_cell.value,
                                    );
                                    app.cursor_row -= 1;
                                }
                            }
                            KeyCode::Char('j') => {
                                app.cursor_row = (app.cursor_row + 1).min(app.rows - 1);
                            }
                            KeyCode::Char('J') => {
                                // moves cell down
                                if app.cursor_row < app.rows - 1 {
                                    app.snapshot();
                                    let (above, below) = app.cells.split_at_mut(app.cursor_row + 1);
                                    let current_cell = &mut above[app.cursor_row][app.cursor_col];
                                    let below_cell = &mut below[0][app.cursor_col];
                                    std::mem::swap(
                                        &mut current_cell.value,
                                        &mut below_cell.value,
                                    );
                                    app.cursor_row += 1;
                                }
                            }
                            KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                app.mode = Mode::VisualColumn;
                                app.visual_start = Some((app.cursor_row, app.cursor_col));
                                app.visual_mode_for_command = Some(app.mode.clone());
                            }
                            KeyCode::Char('v') => {
                                app.mode = Mode::VisualBlock;
                                app.visual_start = Some((app.cursor_row, app.cursor_col));
                                app.visual_mode_for_command = Some(app.mode.clone());
                            }

                            KeyCode::Char('V') => {
                                app.mode = Mode::VisualRow;
                                app.visual_start = Some((app.cursor_row, app.cursor_col));
                                app.visual_mode_for_command = Some(app.mode.clone());
                            }
                            KeyCode::Esc => {
                                app.mode = Mode::Normal;
                                app.visual_start = None;
                            }
                            KeyCode::Char('i') => {
                                app.mode = Mode::Insert;
                                app.input =
                                    app.cells[app.cursor_row][app.cursor_col].to_content();
                                app.insert_mode_cursor = app.input.len();
                            }
                            KeyCode::Char('u') => app.undo(),
                            KeyCode::Char('U') => app.redo(),
                            KeyCode::Char(':') => {
                                app.input.clear();
                                app.mode = Mode::Command;
                            }
                            KeyCode::Char('y') => {
                                app.clipboard =
                                    vec![vec![app.cells[app.cursor_row][app.cursor_col].clone()]];
                            }
                            KeyCode::Char('p') => {
                                if !app.clipboard.is_empty() {
                                    app.snapshot();
                                    let rows = app.clipboard.len();
                                    let cols = app.clipboard[0].len();
                                    for dr in 0..rows {
                                        for dc in 0..cols {
                                            let r = app.cursor_row + dr;
                                            let c = app.cursor_col + dc;
                                            if r < app.rows && c < app.cols {
                                                app.cells[r][c].value =
                                                    app.clipboard[dr][dc].value.clone();
                                            }
                                        }
                                    }
                                }
                            }
                            KeyCode::Char('c') => {
                                app.snapshot();
                                app.mode = Mode::Insert;
                                app.input.clear();
                                app.cells[app.cursor_row][app.cursor_col].value = CellValue::Text("".to_string());
                                app.input = app.cells[app.cursor_row][app.cursor_col].to_content();
                            }
                            KeyCode::Char('d') => {
                                app.snapshot();
                                app.clipboard =
                                    vec![vec![app.cells[app.cursor_row][app.cursor_col].clone()]];
                                app.cells[app.cursor_row][app.cursor_col].value = CellValue::Text("".to_string());
                            }
                            KeyCode::Char('<') => {
                                app.snapshot();
                                app.cwidth = app.cwidth.saturating_sub(1);
                            }
                            KeyCode::Char('>') => {
                                app.snapshot();
                                app.cwidth += 1;
                                if app.cwidth > 20 {
                                    app.cwidth = 20; // Limit max width
                                }
                            }
                            // w/W will page right/left in the spreadsheet
                            KeyCode::Char('e') if key.modifiers.is_empty() => {
                                // Page right: Scroll view right by one page, move cursor to start of new view.
                                if app.view_cols > 0 {
                                    // Calculate the maximum possible scroll_col value to prevent scrolling beyond content
                                    let max_scroll_col = app.cols.saturating_sub(app.view_cols);
                                    app.scroll_col =
                                        (app.scroll_col + app.view_cols).min(max_scroll_col);
                                }
                                // Set cursor to the new scroll position, clamped to valid column index
                                app.cursor_col = app.scroll_col.min(app.cols.saturating_sub(1));
                            }
                            KeyCode::Char('w') if key.modifiers.is_empty() => {
                                // Page left: Scroll view left by one page, move cursor to start of new view.
                                if app.view_cols > 0 {
                                    app.scroll_col = app.scroll_col.saturating_sub(app.view_cols);
                                }
                                // Set cursor to the new scroll position, clamped to valid column index
                                // scroll_col is already >= 0 due to saturating_sub.
                                app.cursor_col = app.scroll_col.min(app.cols.saturating_sub(1));
                            }
                            // b/B will page down/up in the spreadsheet
                            KeyCode::Char('b') if key.modifiers.is_empty() => {
                                // Page down: Scroll view down by one page, move cursor to start of new view.
                                if app.view_rows > 0 {
                                    // Calculate the maximum possible scroll_row value
                                    let max_scroll_row = app.rows.saturating_sub(app.view_rows);
                                    app.scroll_row =
                                        (app.scroll_row + app.view_rows).min(max_scroll_row);
                                }
                                // Set cursor to the new scroll position, clamped to valid row index
                                app.cursor_row = app.scroll_row.min(app.rows.saturating_sub(1));
                            }
                            KeyCode::Char('t') if key.modifiers.is_empty() => {
                                // Page up: Scroll view up by one page, move cursor to start of new view.
                                if app.view_rows > 0 {
                                    app.scroll_row = app.scroll_row.saturating_sub(app.view_rows);
                                }
                                // Set cursor to the new scroll position, clamped to valid row index
                                // scroll_row is already >= 0 due to saturating_sub.
                                app.cursor_row = app.scroll_row.min(app.rows.saturating_sub(1));
                            }
                            _ => {} // This can be kept if truly all other normal mode keys do nothing.
                                    // Or removed if specific fallbacks are preferred. For now, keeping it.
                        },
                        Mode::Insert => match key.code {
                            KeyCode::Esc => app.mode = Mode::Normal,
                            KeyCode::Enter => {
                                app.snapshot();
                                let input = app.input.clone();
                                let cell = if input.starts_with('=') {
                                    Cell { value: CellValue::Formula(input[1..].to_string()) }
                                } else if let Ok(n) = input.parse::<f64>() {
                                    Cell { value: CellValue::Number(n) }
                                } else {
                                    Cell { value: CellValue::Text(input) }
                                };
                                app.cells[app.cursor_row][app.cursor_col] = cell;
                                app.input.clear();
                                app.mode = Mode::Normal;
                                app.command_msg.clear();
                            }
                            KeyCode::Backspace => {
                                app.input.pop();
                            }
                            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                app.input.clear();
                            }
                            KeyCode::Char(c) => {
                                // Insert at the current cursor position in insert mode
                                let pos = app.insert_mode_cursor.min(app.input.len());
                                app.input.insert(pos, c);
                                app.insert_mode_cursor += 1;
                            },
                            _ => {}
                        },
                        Mode::Command => match key.code {
                            KeyCode::Esc => {
                                app.input.clear();
                                if let Some(prev_visual_mode) = app.visual_mode_for_command.clone()
                                {
                                    app.mode = prev_visual_mode;
                                    // visual_start should still be Some.
                                    // visual_mode_for_command should also remain Some.
                                } else {
                                    app.mode = Mode::Normal;
                                    app.visual_start = None;
                                }
                                app.command_history_index = None; // Reset history navigation
                                app.current_command_input_buffer.clear();
                            }
                            KeyCode::Enter => {
                                let command_string = app.input.trim().to_string();
                                if !command_string.is_empty() {
                                    // Add to history if it's not a duplicate of the last command
                                    if app.command_history.last() != Some(&command_string) {
                                        app.command_history.push(command_string.clone());
                                    }
                                }
                                app.command_history_index = None; // Reset history navigation
                                app.current_command_input_buffer.clear();

                                match handle_command(&mut app, &command_string) {
                                    CommandStatus::Success(msg) => {
                                        app.command_msg = msg;
                                    }
                                    CommandStatus::Exit => return Ok(()),
                                    CommandStatus::Error(err) => {
                                        app.command_msg = format!("Error: {}", err);
                                    }
                                }
                                app.input.clear();
                                app.mode = Mode::Normal;
                                app.visual_start = None;
                            }
                            KeyCode::Char(c) => {
                                // Insert at the current cursor position in insert mode
                                let pos = app.insert_mode_cursor.min(app.input.len());
                                app.input.insert(pos, c);
                                app.insert_mode_cursor += 1;
                            },
                            KeyCode::Backspace => {
                                app.input.pop();
                                // if its empty go back to normal mode
                                if app.input.is_empty() {
                                    app.mode = Mode::Normal;
                                    app.visual_start = None;
                                    app.command_history_index = None; // Reset history navigation
                                    app.current_command_input_buffer.clear();
                                }
                            }
                            KeyCode::Up => {
                                if app.command_history.is_empty() {
                                    continue;
                                }
                                match app.command_history_index {
                                    None => {
                                        // Entering history navigation from a new input
                                        app.current_command_input_buffer = app.input.clone();
                                        let last_idx = app.command_history.len() - 1;
                                        app.command_history_index = Some(last_idx);
                                        app.input = app.command_history[last_idx].clone();
                                    }
                                    Some(idx) => {
                                        if idx > 0 {
                                            let new_idx = idx - 1;
                                            app.command_history_index = Some(new_idx);
                                            app.input = app.command_history[new_idx].clone();
                                        } else {
                                            // Already at the oldest command, do nothing or beep
                                        }
                                    }
                                }
                            }
                            KeyCode::Down => {
                                match app.command_history_index {
                                    None => {
                                        // Not in history navigation, do nothing or beep
                                    }
                                    Some(idx) => {
                                        if idx < app.command_history.len() - 1 {
                                            let new_idx = idx + 1;
                                            app.command_history_index = Some(new_idx);
                                            app.input = app.command_history[new_idx].clone();
                                        } else {
                                            // Navigated past the newest command, restore buffer
                                            app.command_history_index = None;
                                            app.input = app.current_command_input_buffer.clone();
                                        }
                                    }
                                }
                            }
                            KeyCode::Tab => {
                                let current_input = app.input.clone();
                                let parts: Vec<&str> = current_input.split_whitespace().collect();

                                if !parts.is_empty() {
                                    let cmd_token = parts[0];
                                    if (cmd_token == "load" || cmd_token == "save")
                                        && (parts.len() == 1
                                            || (!parts.is_empty() && !current_input.ends_with(' ')))
                                    {
                                        let path_to_complete = if parts.len() > 1 {
                                            // Removed mut
                                            parts[1..].join(" ")
                                        } else {
                                            "".to_string()
                                        };

                                        app.tab_completion_command_prefix =
                                            format!("{} ", cmd_token);

                                        let mut base_dir_path = PathBuf::from(".");
                                        let mut partial_item_name = path_to_complete.clone();

                                        if let Some(slash_pos) = path_to_complete.rfind('/') {
                                            base_dir_path =
                                                PathBuf::from(&path_to_complete[..=slash_pos]);
                                            app.tab_completion_path_prefix =
                                                path_to_complete[..=slash_pos].to_string();
                                            partial_item_name =
                                                String::from(&path_to_complete[slash_pos + 1..]);
                                        } else {
                                            app.tab_completion_path_prefix = "".to_string(); // Completing in current dir root
                                        }

                                        if let Ok(entries) = fs::read_dir(&base_dir_path) {
                                            let mut candidates: Vec<String> = entries
                                                .filter_map(Result::ok)
                                                .map(|e| {
                                                    e.file_name().into_string().unwrap_or_default()
                                                })
                                                .filter(|name| {
                                                    name.starts_with(&partial_item_name)
                                                        && name != "."
                                                        && name != ".."
                                                })
                                                .collect();

                                            candidates.sort_by(|a, b| {
                                                let a_is_dir = base_dir_path.join(a).is_dir();
                                                let b_is_dir = base_dir_path.join(b).is_dir();
                                                // Sort directories first, then by name
                                                b_is_dir.cmp(&a_is_dir).then_with(|| a.cmp(b))
                                            });

                                            if !candidates.is_empty() {
                                                if candidates.len() == 1
                                                    && candidates[0] == partial_item_name
                                                {
                                                    // Exact match, try to complete further or add trailing slash/space
                                                    let mut completed_input =
                                                        app.tab_completion_command_prefix.clone();
                                                    completed_input
                                                        .push_str(&app.tab_completion_path_prefix);
                                                    completed_input.push_str(&candidates[0]);
                                                    if base_dir_path.join(&candidates[0]).is_dir() {
                                                        completed_input.push('/');
                                                    }
                                                    app.input = completed_input;
                                                    app.popup_menu = None;
                                                } else {
                                                    app.popup_menu = Some(PopupMenu::new(
                                                        "Suggestions".to_string(),
                                                        candidates,
                                                        10, // Max items visible in popup
                                                    ));
                                                }
                                            } else {
                                                app.popup_menu = None; // No candidates
                                            }
                                        } else {
                                            app.popup_menu = None; // Error reading dir or dir doesn't exist
                                        }
                                    }
                                }
                            }
                            _ => {}
                        },
                        Mode::VisualRow | Mode::VisualBlock | Mode::VisualColumn => {
                            // store selection range for visual modes
                            app.visual_mode_for_command = Some(app.mode.clone());

                            match key.code {
                                // r for result into selection region
                                KeyCode::Char('r') if key.modifiers.is_empty() => {
                                    if app.result.is_some() {
                                        app.snapshot();
                                        if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                                            for r in r1..=r2 {
                                                for c in c1..=c2 {
                                                    app.cells[r][c].value =
                                                        CellValue::Number(app.result.unwrap());
                                                }
                                            }
                                            app.command_msg = format!(
                                                "Result inserted into selection ({}, {}) to ({}, {})",
                                                r1, c1, r2, c2
                                            );
                                        } else {
                                            app.command_msg = "No selection to insert result.".to_string();
                                        }
                                    } else {
                                        app.command_msg = "No result to insert.".to_string();
                                    }
                                }
                                KeyCode::Char('v')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    // Toggle visual column mode
                                    app.mode = Mode::VisualColumn;
                                    if app.visual_start.is_none() {
                                        app.visual_start = Some((app.cursor_row, app.cursor_col));
                                    }
                                    app.visual_mode_for_command = Some(app.mode.clone());
                                }
                                KeyCode::Char('v') => {
                                    // Toggle visual to block mode
                                    app.mode = Mode::VisualBlock;
                                    if app.visual_start.is_none() {
                                        app.visual_start = Some((app.cursor_row, app.cursor_col));
                                    }
                                    app.visual_mode_for_command = Some(app.mode.clone());
                                }
                                KeyCode::Char('V') => {
                                    // Toggle visual row mode
                                    app.mode = Mode::VisualRow;
                                    if app.visual_start.is_none() {
                                        app.visual_start = Some((app.cursor_row, app.cursor_col));
                                    }
                                    app.visual_mode_for_command = Some(app.mode.clone());
                                }
                                KeyCode::Char('b') => {
                                    // Toggle visual block mode
                                    app.mode = Mode::VisualBlock;
                                    if app.visual_start.is_none() {
                                        app.visual_start = Some((app.cursor_row, app.cursor_col));
                                    }
                                    app.visual_mode_for_command = Some(app.mode.clone());
                                }
                                KeyCode::Char('c') => {
                                    // Toggle visual column mode
                                    app.mode = Mode::VisualColumn;
                                    if app.visual_start.is_none() {
                                        app.visual_start = Some((app.cursor_row, app.cursor_col));
                                    }
                                    app.visual_mode_for_command = Some(app.mode.clone());
                                }
                                // Other visual mode operations would continue here
                                KeyCode::Esc => {
                                    app.mode = Mode::Normal;
                                    app.visual_start = None;
                                }
                                KeyCode::Char(':') => {
                                    app.input.clear();
                                    app.visual_mode_for_command = Some(app.mode.clone());
                                    app.mode = Mode::Command;
                                }
                                KeyCode::Char('h') => {
                                    app.cursor_col = app.cursor_col.saturating_sub(1)
                                }
                                KeyCode::Char('l') => {
                                    app.cursor_col = (app.cursor_col + 1).min(app.cols - 1)
                                }
                                KeyCode::Char('k') => {
                                    app.cursor_row = app.cursor_row.saturating_sub(1)
                                }
                                KeyCode::Char('j') => {
                                    app.cursor_row = (app.cursor_row + 1).min(app.rows - 1)
                                }
                                // Move the entire selected range with Shift+H/L/K/J
                                KeyCode::Char('H') => {
                                    if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                                        if c1 > 0 {
                                            app.snapshot();
                                            for r in r1..=r2 {
                                                let row = &mut app.cells[r];
                                                for c in c1..=c2 {
                                                    // Use split_at_mut to get two non-overlapping mutable references
                                                    let (left, right) = row.split_at_mut(c);
                                                    std::mem::swap(
                                                        &mut left[c - 1].value,
                                                        &mut right[0].value,
                                                    );
                                                }
                                            }
                                            app.cursor_col = app.cursor_col.saturating_sub(1);
                                            app.visual_start = app
                                                .visual_start
                                                .map(|(vr, vc)| (vr, vc.saturating_sub(1)));
                                        }
                                    }
                                }
                                KeyCode::Char('L') => {
                                    if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                                        if c2 < app.cols - 1 {
                                            app.snapshot();
                                            for r in r1..=r2 {
                                                let row = &mut app.cells[r];
                                                for c in (c1..=c2).rev() {
                                                    // Use split_at_mut to get two non-overlapping mutable references
                                                    let (left, right) = row.split_at_mut(c + 1);
                                                    std::mem::swap(&mut left[c], &mut right[0]);
                                                }
                                            }
                                            app.cursor_col = (app.cursor_col + 1).min(app.cols - 1);
                                            app.visual_start = app
                                                .visual_start
                                                .map(|(vr, vc)| (vr, (vc + 1).min(app.cols - 1)));
                                        }
                                    }
                                }
                                KeyCode::Char('K') => {
                                    if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                                        if r1 > 0 {
                                            app.snapshot();
                                            for r in r1..=r2 {
                                                // Use split_at_mut to get two non-overlapping mutable references to rows
                                                let (top, bottom) = app.cells.split_at_mut(r);
                                                let above_row = &mut top[r - 1];
                                                let current_row = &mut bottom[0];
                                                for c in c1..=c2 {
                                                    std::mem::swap(
                                                        &mut above_row[c].value,
                                                        &mut current_row[c].value,
                                                    );
                                                }
                                            }
                                            app.cursor_row = app.cursor_row.saturating_sub(1);
                                            app.visual_start = app
                                                .visual_start
                                                .map(|(vr, vc)| (vr.saturating_sub(1), vc));
                                        }
                                    }
                                }
                                KeyCode::Char('J') => {
                                    if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                                        if r2 < app.rows - 1 {
                                            app.snapshot();
                                            for r in (r1..=r2).rev() {
                                                // Use split_at_mut to get two non-overlapping mutable references to rows
                                                let (top, bottom) = app.cells.split_at_mut(r + 1);
                                                let current_row = &mut top[r];
                                                let below_row = &mut bottom[0];
                                                for c in c1..=c2 {
                                                    std::mem::swap(
                                                        &mut current_row[c].value,
                                                        &mut below_row[c].value,
                                                    );
                                                }
                                            }
                                            app.cursor_row = (app.cursor_row + 1).min(app.rows - 1);
                                            app.visual_start = app
                                                .visual_start
                                                .map(|(vr, vc)| ((vr + 1).min(app.rows - 1), vc));
                                        }
                                    }
                                }
                                KeyCode::Char('y') => {
                                    if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                                        app.clipboard.clear();
                                        for r in r1..=r2 {
                                            for c in c1..=c2 {
                                                if app.clipboard.len() <= r - r1 {
                                                    app.clipboard.push(Vec::new());
                                                }
                                                app.clipboard[r - r1].push(app.cells[r][c].clone());
                                            }
                                        }
                                        app.mode = Mode::Normal;
                                        app.visual_start = None;
                                    }
                                }
                                KeyCode::Char('p') => {
                                    if !app.clipboard.is_empty() {
                                        app.snapshot();
                                        if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                                            for (dr, r) in (r1..=r2).enumerate() {
                                                if dr < app.clipboard.len() {
                                                    for (dc, c) in (c1..=c2).enumerate() {
                                                        if dc < app.clipboard[dr].len() {
                                                            app.cells[r][c].value = app.clipboard
                                                                [dr][dc]
                                                                .value
                                                                .clone();
                                                        }
                                                    }
                                                }
                                            }
                                            app.mode = Mode::Normal;
                                            app.visual_start = None;
                                        } else if !app.clipboard.is_empty()
                                            && !app.clipboard[0].is_empty()
                                        {
                                            app.cells[app.cursor_row][app.cursor_col].value =
                                                app.clipboard[0][0].value.clone();
                                        }
                                    }
                                }
                                KeyCode::Char('d') => {
                                    app.snapshot();
                                    if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                                        // Copy selection to clipboard
                                        app.clipboard.clear();
                                        for r in r1..=r2 {
                                            let mut row_clip = Vec::new();
                                            for c in c1..=c2 {
                                                row_clip.push(app.cells[r][c].clone());
                                                app.cells[r][c].value = CellValue::Text("".to_string());
                                            }
                                            app.clipboard.push(row_clip);
                                        }
                                        app.mode = Mode::Normal;
                                      app.visual_start = None;
                                    } else {
                                        // Copy single cell to clipboard
                                        app.clipboard = vec![vec![
                                            app.cells[app.cursor_row][app.cursor_col].clone(),
                                        ]];
                                        app.cells[app.cursor_row][app.cursor_col].value = CellValue::Text("".to_string());
                                    }
                                }
                                _ => {}
                            }

                            if app.last_selection_range.is_none() {
                                app.last_selection_range = app.selected_range();
                            } else {
                                // Update the last selection range if it exists
                                if let Some((start_row, start_col)) = app.visual_start {
                                    app.last_selection_range = Some((
                                        (start_row, start_col),
                                        (app.cursor_row, app.cursor_col),
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Vertical scroll
                if app.view_rows > 0 {
                    if app.cursor_row < app.scroll_row {
                        // Cursor is above the visible area, scroll up
                        app.scroll_row = app.cursor_row;
                    } else if app.cursor_row >= app.scroll_row + app.view_rows {
                        // Cursor is at or below the last visible row, scroll down
                        app.scroll_row = app.cursor_row - app.view_rows + 1;
                    }
                } else {
                    // No viewable rows, try to keep scroll_row consistent with cursor_row
                    app.scroll_row = app.cursor_row;
                }

                // Horizontal scroll
                if app.view_cols > 0 {
                    if app.cursor_col < app.scroll_col {
                        // Cursor is to the left of the visible area, scroll left
                        app.scroll_col = app.cursor_col;
                    } else if app.cursor_col >= app.scroll_col + app.view_cols {
                        // Cursor is at or to the right of the last visible column, scroll right
                        app.scroll_col = app.cursor_col - app.view_cols + 1;
                    }
                } else {
                    // No viewable columns, try to keep scroll_col consistent with cursor_col
                    app.scroll_col = app.cursor_col;
                }

                // Clamp scroll values to ensure they are within valid ranges
                if app.rows > 0 {
                    app.scroll_row = app.scroll_row.min(app.rows.saturating_sub(app.view_rows));
                } else {
                    app.scroll_row = 0; // If no rows, scroll_row must be 0
                }
                // app.scroll_row is usize, so it's implicitly >= 0.

                if app.cols > 0 {
                    app.scroll_col = app.scroll_col.min(app.cols.saturating_sub(app.view_cols));
                } else {
                    app.scroll_col = 0; // If no cols, scroll_col must be 0
                }
                // app.scroll_col is usize, so it's implicitly >= 0.
            }
        }
    }
}

// SelectionCommand trait and implementations

trait SelectionCommand {
    fn name(&self) -> &'static str;
    fn modifies_data(&self) -> bool;
    fn execute(
        &self,
        app: &mut App,
        r1: usize,
        c1: usize,
        r2: usize,
        c2: usize,
    ) -> Result<String, String>;
}

struct UppercaseCommand;
struct LowercaseCommand;
struct SumCommand;
struct AverageCommand;

impl SelectionCommand for UppercaseCommand {
    fn name(&self) -> &'static str {
        "uppercase"
    }
    fn modifies_data(&self) -> bool {
        true
    }
    fn execute(
        &self,
        app: &mut App,
        r1: usize,
        c1: usize,
        r2: usize,
        c2: usize,
    ) -> Result<String, String> {
        for r in r1..=r2 {
            for c in c1..=c2 {
                app.cells[r][c].value = CellValue::Text(app.cells[r][c].to_content().to_uppercase());
            }
        }
        Ok("Converted selection to uppercase.".to_string())
    }
}

impl SelectionCommand for LowercaseCommand {
    fn name(&self) -> &'static str {
        "lowercase"
    }
    fn modifies_data(&self) -> bool {
        true
    }
    fn execute(
        &self,
        app: &mut App,
        r1: usize,
        c1: usize,
        r2: usize,
        c2: usize,
    ) -> Result<String, String> {
        for r in r1..=r2 {
            for c in c1..=c2 {
                app.cells[r][c].value = CellValue::Text(app.cells[r][c].to_content().to_lowercase());
            }
        }
        Ok("Converted selection to lowercase.".to_string())
    }
}

impl SelectionCommand for SumCommand {
    fn name(&self) -> &'static str {
        "sum"
    }
    fn modifies_data(&self) -> bool {
        false
    }
    fn execute(
        &self,
        app: &mut App,
        r1: usize,
        c1: usize,
        r2: usize,
        c2: usize,
    ) -> Result<String, String> {
        use crate::CellValue::*;
        use crate::parse_expr;
        use crate::eval_expr;
        use std::collections::HashMap;
        let mut sum = 0.0;
        let mut count = 0;
        // Build cell map for formula evaluation
        let mut cell_map: HashMap<(usize, usize), &CellValue> = HashMap::new();
        for (r_idx, row_vec) in app.cells.iter().enumerate() {
            for (c_idx, cell_val) in row_vec.iter().enumerate() {
                cell_map.insert((r_idx, c_idx), &cell_val.value);
            }
        }
        for r in r1..=r2 {
            for c in c1..=c2 {
                match &app.cells[r][c].value {
                    Number(n) => {
                        sum += *n;
                        count += 1;
                    }
                    Formula(expr_str) => {
                        if let Some(expr) = parse_expr(expr_str) {
                            let mut visited = std::collections::HashSet::new();
                            let val = eval_expr(&expr, &cell_map, &mut visited);
                            sum += val;
                            count += 1;
                        }
                    }
                    Text(s) => {
                        if let Ok(n) = s.trim().parse::<f64>() {
                            sum += n;
                            count += 1;
                        }
                    }
                }
            }
        }
        if count == 0 {
            Err("No numeric or formula values in selection.".to_string())
        } else {
            let result = sum;
            app.result = Some(result);
            Ok(format!("Sum: {}", sum))
        }
    }
}

impl SelectionCommand for AverageCommand {
    fn name(&self) -> &'static str {
        "average"
    }
    fn modifies_data(&self) -> bool {
        false
    }
    fn execute(
        &self,
        app: &mut App,
        r1: usize,
        c1: usize,
        r2: usize,
        c2: usize,
    ) -> Result<String, String> {
        use crate::CellValue::*;
        use crate::parse_expr;
        use crate::eval_expr;
        use std::collections::HashMap;
        let mut sum = 0.0;
        let mut count = 0;
        // Build cell map for formula evaluation
        let mut cell_map: HashMap<(usize, usize), &CellValue> = HashMap::new();
        for (r_idx, row_vec) in app.cells.iter().enumerate() {
            for (c_idx, cell_val) in row_vec.iter().enumerate() {
                cell_map.insert((r_idx, c_idx), &cell_val.value);
            }
        }
        for r in r1..=r2 {
            for c in c1..=c2 {
                match &app.cells[r][c].value {
                    Number(n) => {
                        sum += *n;
                        count += 1;
                    }
                    Formula(expr_str) => {
                        if let Some(expr) = parse_expr(expr_str) {
                            let mut visited = std::collections::HashSet::new();
                            let val = eval_expr(&expr, &cell_map, &mut visited);
                            sum += val;
                            count += 1;
                        }
                    }
                    Text(s) => {
                        if let Ok(n) = s.trim().parse::<f64>() {
                            sum += n;
                            count += 1;
                        }
                    }
                }
            }
        }
        if count == 0 {
            Err("No numeric or formula values in selection.".to_string())
        } else {
            let result = sum / count as f64;
            app.result = Some(result);
            Ok(format!("Average: {}", result))
        }
    }
}

fn get_selection_command(name: &str) -> Option<Box<dyn SelectionCommand>> {
    match name {
        "uppercase" => Some(Box::new(UppercaseCommand)),
        "lowercase" => Some(Box::new(LowercaseCommand)),
        "sum" => Some(Box::new(SumCommand)),
        "average" => Some(Box::new(AverageCommand)),
        _ => None,
    }
}

// Add a help message constant for the help popup
const HELP_MESSAGE: &str = r#"Spreadsheet TUI Help

General:
    Ctrl+q          Quit application immediately
    Esc             Exit visual/insert/command mode, or close popups

Navigation (Normal Mode):
    h/j/k/l         Move cursor left/down/up/right
    0/$             Move cursor to first/last column of the current row
    g/G             Jump cursor to the first/last row of the sheet
    w/e             Page view left/right (cursor moves to start of new view)
    t/b             Page view up/down (cursor moves to start of new view)
    ,               Restore last visual selection (re-enters visual mode)

Editing (Normal Mode):
    r               Enter Command mode to operate on current cell (e.g., :sum)
    i               Enter Insert mode to edit current cell (pre-fills with cell content)
    f               Enter Insert mode to edit current cell as a formula (pre-fills with '=')
    c               Clear current cell content and enter Insert mode
    d               Delete current cell content (copies to clipboard)
    D               Delete row
    ctrl+d          Delete column
    o               Insert new row below current row
    O               Insert new row above current row
    a               Append new column to the right of current column
    A               Append new column to the left of current column
    y               Yank (copy) current cell content to clipboard
    p               Paste clipboard content starting at cursor (can paste multiple cells)
    H/J/K/L         Move current cell content left/down/up/right by swapping with adjacent cell

Visual Mode (Enter with V, v, Ctrl+v from Normal Mode):
    V               Enter Visual Row selection mode
    v               Enter Visual Block selection mode
    Ctrl+v          Enter Visual Column selection mode
    r               Insert result into selected cells (if available)
    In Visual Mode:
    h/j/k/l         Adjust selection boundary
    H/J/K/L         Move entire selected block of cells left/down/up/right
    y               Yank (copy) selected cells to clipboard
    d               Delete selected cells (copies to clipboard) and clear their content
    p               Paste clipboard content into the selected region
    :               Enter Command mode to operate on selection (e.g., :sum, :sort)
    Esc             Exit Visual mode, return to Normal mode
    V/v/b/c         Switch between Visual Row (V), Visual Block (v or b), Visual Column (c or Ctrl+v)

Insert Mode (Enter with i, f, c from Normal Mode):
    Type to edit cell content.
    Enter           Save changes to cell and return to Normal mode
    Esc             Discard changes and return to Normal mode
    Backspace       Delete character before cursor
    Ctrl+u          Clear entire input line
    (Arrow keys for cursor movement within input are not supported)

Command Mode (Enter with : from Normal or Visual Mode):
    Type command and press Enter. Esc to cancel.
    Up/Down         Navigate command history
    Tab             Attempt path completion for :load, :save, :w, :wq

    Commands:
    :q              Quit
    :w [<file>]     Write (save) sheet to CSV. Uses current filepath if <file> omitted.
    :wq [<file>]    Write (save) sheet and quit.
    :save <file>    Save sheet to <file> as CSV.
    :load <file>    Load sheet from <file>.csv.
    :files          Show files in current directory (popup).
    :help           Show this help message (popup).
    :goto <r>[:<c>] Jump cursor to row <r> and optional column <c> (0-indexed).
    :r <row>        Jump cursor to <row> (0-indexed).
    :c <col>        Jump cursor to <col> (0-indexed).
    :clear          Clear all cells in the sheet.
    :undo           Undo last action.
    :redo           Redo last undone action.
    :num / :number  Convert text cells to numbers if parseable. Formulas/numbers unchanged.
    :fill [content] Fill selection. Uses [content] if provided. If no content, uses the
                    first cell of the clipboard.
    :new (r|c|rc) [N] Insert N (default 1) new row(s) at cursor/selection (r=row, c=col, rc=both).
    :delete (r|c|rc) Delete row(s)/column(s) at cursor/selection.

    Selection Commands (typically used from Visual Mode with :):
    :uppercase      Convert text in selection to uppercase.
    :lowercase      Convert text in selection to lowercase.
    :sum            Calculate sum of numeric values in selection (displays in command message).
    :average        Calculate average of numeric values in selection (displays in command message).
    :sort [flags]   Sort selected cells. Behavior depends on Visual Mode:
                    - Visual Row/Block: Sorts rows within the selection.
                    - Visual Column: Sorts all rows of the sheet based on the selected columns.
      Flags:
        n,num,numeric   Sort numerically (default).
        s,str,string    Sort as strings.
        l,len,length    Sort by content length.
        <,asc           Ascending order (default).
        >,desc          Descending order.
        e,ext,extended  Extended sort:
                        - Visual Row/Block: Moves the entire content of selected rows.
                        - Visual Column: Sorts all rows of the sheet based on the key column.

Formulas:
    Start cell content with '=' (e.g., in Insert mode or via 'f' key).
    Example: =A1+B2 or =(A1+B2)*C3
    Supported:
      Operators: +, -, *, / and parentheses ().
      Cell Refs: A1, B10 (0-indexed internally, e.g. A0 is (0,0)).
      Ranges: A1:C5 (e.g., for SUM).
      Functions: SUM(range), AVERAGE(args), COUNT(args), MAX(args), MIN(args).
                 Arguments can be numbers, cell refs, or ranges.
    Evaluation:
      Non-numeric cells or cells with errors referenced in formulas are often treated as 0.0
      or may result in NaN (Not a Number), affecting calculations.
      Circular references result in NaN.
      Inspector bar (below sheet) shows the current cell's raw content and, if a formula,
      its evaluated result (e.g., =A1+B1 → 15.0).
      Errors like #DIV/0!, #INF!, #P_ERR (parse error) may appear.

Undo/Redo:
    u               Undo last action (Normal mode)
    U               Redo last undone action (Normal mode, Shift+u)

Column Width:
    < / >           Decrease/increase display width of all columns (Normal mode)

Help Popup Navigation:
    j/k             Scroll down/up
    g               Scroll to top
    Esc, q, h       Close help

Press Esc, q, or h to close this help.
"#;
