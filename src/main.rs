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
    collections::VecDeque,
    env,
    fs,
    fs::File,
    io,
    path::PathBuf, // Removed unused Path
    time::Duration,
}; // Added env

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
struct Cell {
    content: String,
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
}

impl App {
    pub fn new(rows: usize, cols: usize) -> Self {
        let cwidth = 10;
        let cells = vec![
            vec![
                Cell {
                    content: "".to_string()
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
                    .content
                    .clone()
            });
            if params.sort_order == SortOrder::Descending {
                row_indices_to_sort.reverse();
            }
            row_indices_to_sort.sort_by(|&row_a_idx, &row_b_idx| {
                let val_a = &app.cells[row_a_idx]
                    [sort_key_row_index.min(app.cols.saturating_sub(1))]
                .content;
                let val_b = &app.cells[row_b_idx]
                    [sort_key_row_index.min(app.cols.saturating_sub(1))]
                .content;
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
                    .content
                    .clone()
            });
            if params.sort_order == SortOrder::Descending {
                row_indices_to_sort.reverse();
            }
            row_indices_to_sort.sort_by(|&row_a_idx, &row_b_idx| {
                let val_a = &app.cells[row_a_idx]
                    [sort_key_column_index.min(app.cols.saturating_sub(1))]
                .content;
                let val_b = &app.cells[row_b_idx]
                    [sort_key_column_index.min(app.cols.saturating_sub(1))]
                .content;
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
                    .content
                    .clone()
            });
            if params.sort_order == SortOrder::Descending {
                row_indices_to_sort.reverse();
            }
            row_indices_to_sort.sort_by(|&row_a_idx, &row_b_idx| {
                let val_a = &app.cells[row_a_idx]
                    [sort_key_column_index.min(app.cols.saturating_sub(1))]
                .content;
                let val_b = &app.cells[row_b_idx]
                    [sort_key_column_index.min(app.cols.saturating_sub(1))]
                .content;
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
                    app.cells[r][c].content = content_arg.to_string();
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
            let clipboard_content_to_fill = app.clipboard[0][0].content.clone();
            if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                app.snapshot(); // Save current state before filling
                for r in r1..=r2 {
                for c in c1..=c2 {
                    if r < app.rows && c < app.cols {
                    app.cells[r][c].content = clipboard_content_to_fill.clone();
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
                        content: "".to_string()
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
            if !app.cells[r][c].content.is_empty() {
                row_has_data = true;
                actual_cols = actual_cols.max(c + 1);
            }
        }
        if row_has_data {
            actual_rows = r + 1;
        }
    }

    for r_idx in 0..actual_rows {
        let record: Vec<&str> = (0..actual_cols)
            .map(|c_idx| {
                // Ensure we are within the bounds of the app's cell structure for safety,
                // though actual_cols should be derived from these bounds.
                if r_idx < app.rows && c_idx < app.cols {
                    app.cells[r_idx][c_idx].content.as_str()
                } else {
                    "" // Should ideally not be reached if logic is correct
                }
            })
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
                    content: "".to_string()
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
                    content: "".to_string()
                };
                app.cols
            ];
            app.rows
        ];

        for (r, row_data) in records_data.iter().enumerate() {
            for (c, field_content) in row_data.iter().enumerate() {
                // These checks should be fine as dimensions are now based on CSV
                if r < app.rows && c < app.cols {
                    app.cells[r][c].content = field_content.clone();
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
            let header = String::new();
            let mut header_spans: Vec<Span> = Vec::new();
            // Add empty span for the space above row numbers
            header_spans.push(Span::raw(format!("{:width$}", " ", width = number_width)));

            for c_idx in app.scroll_col..(app.scroll_col + app.view_cols).min(app.cols) {
                let mut is_highlighted = false;

                // Check if the cursor is in the current column
                if c_idx == app.cursor_col {
                    is_highlighted = true;
                }

                // Check if the current column is part of a visual selection's column range
                if !is_highlighted { // Only check if not already highlighted by the cursor
                    if let Some(((_r1, sel_c1), (_r2, sel_c2))) = app.selected_range() {
                        // app.selected_range() provides the correct column bounds (sel_c1, sel_c2)
                        // based on the current visual mode (VisualRow, VisualColumn, VisualBlock).
                        // For VisualRow, sel_c1 will be 0 and sel_c2 will be app.cols - 1,
                        // effectively highlighting all column headers.
                        if c_idx >= sel_c1 && c_idx <= sel_c2 {
                            is_highlighted = true;
                        }
                    }
                }

                let col_text = format!("{:<width$}", c_idx, width = app.cwidth + 2);
                let style = if is_highlighted {
                    Style::default().add_modifier(Modifier::REVERSED) // Style for highlighted header
                } else {
                    Style::default() // Default style for non-highlighted header
                };
                header_spans.push(Span::styled(col_text, style));
            }
            lines.push(Line::from(header_spans));

            for r in app.scroll_row..(app.scroll_row + app.view_rows).min(app.rows) {
                let mut row_spans: Vec<Span> = Vec::new();
                let mut is_row_highlighted = false;
                // Highlight if cursor is on this row
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
                    Style::default().add_modifier(Modifier::REVERSED) // Style for highlighted row number
                } else {
                    Style::default() // Default style for non-highlighted row number
                };
                row_spans.push(Span::styled(row_number_text, row_number_style));


                for c in app.scroll_col..(app.scroll_col + app.view_cols).min(app.cols) {
                    // Get cell content, truncate, and center it
                    let original_content = if r < app.rows && c < app.cols {
                        app.cells[r][c].content.clone()
                    } else {
                        "".to_string()
                    };

                    let truncated_content = original_content
                        .chars()
                        .take(app.cwidth)
                        .collect::<String>();
                    let centered_content =
                        format!("{:^width$}", truncated_content, width = app.cwidth);

                    // Determine if the cell is the cursor or part of a selection
                    let is_cursor = r == app.cursor_row && c == app.cursor_col;
                    let is_selected = if let Some(((r1, c1), (r2, c2))) = app.selected_range() {
                        // selected_range() already provides the correct bounding box for the current visual mode
                        r >= r1 && r <= r2 && c >= c1 && c <= c2
                    } else {
                        false
                    };

                    let cell_text: String;
                    let cell_style: Style;

                    if is_cursor {
                        cell_text = format!("[{}]", centered_content); // Keep brackets for cursor
                        cell_style = Style::default().add_modifier(Modifier::REVERSED); // Use reverse video for cursor
                    } else if is_selected {
                        cell_text = format!(" {centered_content} "); // Keep asterisks or use spaces
                        // Style for selected cells (e.g., background color)
                        cell_style = Style::default().bg(Color::DarkGray); // Example: Dark gray background
                    // Consider adding .fg(Color::White) if needed for contrast
                    } else {
                        cell_text = format!(" {} ", centered_content); // Spaces for normal cells
                        cell_style = Style::default(); // Default style for normal cells
                    }
                    row_spans.push(Span::styled(cell_text, cell_style));
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
            let current_cell_content = if app.rows > 0 && app.cols > 0 {
                &app.cells[app.cursor_row.min(app.rows - 1)][app.cursor_col.min(app.cols - 1)]
                    .content
            } else {
                "N/A"
            };
            let inspector_text = format!(
                "Cell ({}, {}): \"{}\" | Size: {}Rx{}C",
                app.cursor_row, app.cursor_col, current_cell_content, app.rows, app.cols,
            );
            let inspector_paragraph = Paragraph::new(inspector_text);
            f.render_widget(inspector_paragraph, inspector_area);

            // Footer / Command Bar
            let footer_text = match app.mode {
                Mode::Command => format!(":{}", app.input),
                Mode::Insert => format!("INSERT {}", app.input),
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
                    // Existing key handling logic when help and popup_menu are not shown
                    match app.mode {
                        Mode::Normal => match key.code {
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
                                app.cursor_col = app.cursor_col.saturating_sub(1)
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
                                        &mut left_cell.content,
                                        &mut current_cell.content,
                                    );
                                    app.cursor_col -= 1;
                                }
                            }
                            KeyCode::Char('l') => {
                                app.cursor_col = (app.cursor_col + 1).min(app.cols - 1)
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
                                        &mut current_cell.content,
                                        &mut right_cell.content,
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
                                        &mut current_cell.content,
                                        &mut below_cell.content,
                                    );
                                    app.cursor_row -= 1;
                                }
                            }
                            KeyCode::Char('j') => {
                                app.cursor_row = (app.cursor_row + 1).min(app.rows - 1)
                            }
                            KeyCode::Char('J') => {
                                // moves cell down
                                if app.cursor_row < app.rows - 1 {
                                    app.snapshot();
                                    let (above, below) = app.cells.split_at_mut(app.cursor_row + 1);
                                    let current_cell = &mut above[app.cursor_row][app.cursor_col];
                                    let below_cell = &mut below[0][app.cursor_col];
                                    std::mem::swap(
                                        &mut current_cell.content,
                                        &mut below_cell.content,
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
                                    app.cells[app.cursor_row][app.cursor_col].content.clone();
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
                                                app.cells[r][c].content =
                                                    app.clipboard[dr][dc].content.clone();
                                            }
                                        }
                                    }
                                }
                            }
                            KeyCode::Char('c') => {
                                app.snapshot();
                                app.mode = Mode::Insert;
                                app.input.clear();
                                app.cells[app.cursor_row][app.cursor_col].content.clear();
                            }
                            KeyCode::Char('d') => {
                                app.snapshot();
                                app.clipboard =
                                    vec![vec![app.cells[app.cursor_row][app.cursor_col].clone()]];
                                app.cells[app.cursor_row][app.cursor_col].content.clear();
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
                                app.cells[app.cursor_row][app.cursor_col].content =
                                    app.input.clone();
                                app.input.clear();
                                app.mode = Mode::Normal;
                            }
                            KeyCode::Backspace => {
                                app.input.pop();
                            }

                            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                app.input.clear();
                            }
                            KeyCode::Char(c) => app.input.push(c),
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
                            KeyCode::Char(c) => app.input.push(c),
                            KeyCode::Backspace => {
                                app.input.pop();
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
                                                    app.popup_menu = None; // No need for popup if only one exact match that's already typed
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
                                                        &mut left[c - 1].content,
                                                        &mut right[0].content,
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
                                                        &mut above_row[c].content,
                                                        &mut current_row[c].content,
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
                                                        &mut current_row[c].content,
                                                        &mut below_row[c].content,
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
                                                            app.cells[r][c].content = app.clipboard
                                                                [dr][dc]
                                                                .content
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
                                            app.cells[app.cursor_row][app.cursor_col].content =
                                                app.clipboard[0][0].content.clone();
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
                                                app.cells[r][c].content.clear();
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
                                        app.cells[app.cursor_row][app.cursor_col].content.clear();
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
                app.cells[r][c].content = app.cells[r][c].content.to_uppercase();
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
                app.cells[r][c].content = app.cells[r][c].content.to_lowercase();
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
        let mut sum = 0.0;
        let mut count = 0;
        for r in r1..=r2 {
            for c in c1..=c2 {
                if let Ok(val) = app.cells[r][c].content.trim().parse::<f64>() {
                    sum += val;
                    count += 1;
                }
            }
        }
        if count == 0 {
            Err("No numeric values in selection.".to_string())
        } else {
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
        let mut sum = 0.0;
        let mut count = 0;
        for r in r1..=r2 {
            for c in c1..=c2 {
                if let Ok(val) = app.cells[r][c].content.trim().parse::<f64>() {
                    sum += val;
                    count += 1;
                }
            }
        }
        if count == 0 {
            Err("No numeric values in selection.".to_string())
        } else {
            Ok(format!("Average: {}", sum / count as f64))
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

Navigation:
    h/j/k/l         Move left/down/up/right
    H/J/K/L         Move cell left/down/up/right (swap)
    0/$             Move to first/last column
    g/G             Jump to first/last row
    V               Start visual row selection
    v               Start visual block selection
    Ctrl+v          Start visual column selection
    Esc             Exit visual/insert/command mode

Editing:
    i               Edit cell (insert mode)
    c               Clear cell and enter insert mode
    d               Delete cell or selection (copies to clipboard)
    y               Yank (copy) cell or selection
    p               Paste clipboard at cursor or selection

Undo/Redo:
    u               Undo
    U               Redo

Column Width:
    < / >           Decrease/increase column width

Command Mode:
    :               Enter command mode
    :w <file>       Save as CSV
    :q              Quit
    :wq <file>      Save and quit
    :load <file>    Load CSV file
    :save <file>    Save as CSV
    :files          Show files in current directory
    :help           Show this help
    :goto <r> [: <c>] Jump to specific row/column
    :clear          Clear all cells
    :undo           Undo last action
    :redo           Redo last action

Selection Commands:
    :uppercase      Convert selection to uppercase
    :lowercase      Convert selection to lowercase
    :sum            Calculate sum of numeric values in selection
    :average        Calculate average of numeric values in selection

Sort:
    In visual mode, use :sort [flags]
    Flags:
        n, num, numeric     Sort numerically (default)
        s, str, string      Sort as strings
        l, len, length      Sort by content length
        <, asc             Ascending order (default)
        >, desc            Descending order
        e, ext, extended    Sort entire rows based on selected cells

Visual Mode:
    Move selection with h/j/k/l
    Move entire selection with H/J/K/L
    Apply commands with : (e.g., :sort, :uppercase)
    Switch between visual modes with V (row), v (block), c (column), b (block)

Other:
    Tab completion for :load and :save commands

Press Esc, q, or h to close this help.
"#;

// Helper function to create a centered rectangle for the popup
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

// simple forth script engine to create formulas and attach to cells
