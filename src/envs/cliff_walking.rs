use std::collections::BTreeMap;

use crate::mdp::{self, Mdp, Reward, State, Transition};

#[derive(Copy, Clone, Debug)]
pub(crate) enum Cell {
    Start,
    Regular,
    Cliff,
    End,
}

pub(crate) enum Action {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

pub(crate) const ROWS: usize = 4;
pub(crate) const COLS: usize = 12;
pub(crate) const STEP_REWARD: f64 = -1.0;
pub(crate) const CLIFF_REWARD: f64 = -100.0;
pub(crate) const END_REWARD: f64 = 0.0;

pub fn build_mdp() -> Mdp {
    // create grid of cells
    let mut grid: [[Cell; ROWS]; COLS] = [[Cell::Regular; ROWS]; COLS];

    //set lower row
    grid[0][ROWS - 1] = Cell::Start;
    for cell in grid.iter_mut().take(COLS - 1).skip(1) {
        cell[ROWS - 1] = Cell::Cliff;
    }
    grid[COLS - 1][ROWS - 1] = Cell::End;

    let mut transitions: BTreeMap<(State, mdp::Action), Vec<Transition>> = BTreeMap::new();

    // fill transition map for every state and action
    for row in 0..ROWS {
        for col in 0..COLS {
            let from_state = State(row * COLS + col);

            // Up action
            // can only run into wall with up
            let action = Action::Up;
            if row as isize - 1 < 0 {
                let (key, value) = build_transition(from_state, from_state, action, STEP_REWARD);
                transitions.insert(key, value);
            } else {
                let to_state = State((row - 1) * COLS + col);
                let (key, value) = build_transition(from_state, to_state, action, STEP_REWARD);
                transitions.insert(key, value);
            }

            // Left action
            // can only run into wall with left
            let action = Action::Left;
            if col as isize - 1 < 0 {
                let (key, value) = build_transition(from_state, from_state, action, STEP_REWARD);
                transitions.insert(key, value);
            } else {
                let to_state = State(row * COLS + (col - 1));
                let (key, value) = build_transition(from_state, to_state, action, STEP_REWARD);
                transitions.insert(key, value);
            }

            //Right action
            let action = Action::Right;
            if col + 1 == COLS {
                let (key, value) = build_transition(from_state, from_state, action, STEP_REWARD);
                transitions.insert(key, value);
            } else {
                let to_state = State(row * COLS + (col + 1));
                let (key, value) = match grid[col + 1][row] {
                    Cell::Start => build_transition(from_state, to_state, action, STEP_REWARD),
                    Cell::Regular => build_transition(from_state, to_state, action, STEP_REWARD),
                    Cell::Cliff => build_transition(from_state, to_state, action, CLIFF_REWARD),
                    Cell::End => build_transition(from_state, to_state, action, END_REWARD),
                };
                transitions.insert(key, value);
            }

            //Down action
            let action = Action::Down;
            if row + 1 == ROWS {
                let (key, value) = build_transition(from_state, from_state, action, STEP_REWARD);
                transitions.insert(key, value);
            } else {
                let to_state = State((row + 1) * COLS + col);
                let (key, value) = match grid[col][row + 1] {
                    Cell::Start => build_transition(from_state, to_state, action, STEP_REWARD),
                    Cell::Regular => build_transition(from_state, to_state, action, STEP_REWARD),
                    Cell::Cliff => build_transition(from_state, to_state, action, CLIFF_REWARD),
                    Cell::End => build_transition(from_state, to_state, action, END_REWARD),
                };
                transitions.insert(key, value);
            }
        }
    }

    let mut terminal_states = vec![];

    // last row all terminal except first cell
    for col in 1..COLS {
        terminal_states.push(State((ROWS - 1) * COLS + col));
    }

    print_grid(&grid);
    Mdp {
        transitions,
        terminal_states,
        initial_state: State((ROWS - 1) * COLS),
    }
}

fn build_transition(
    from_state: State,
    to_state: State,
    action: Action,
    reward: Reward,
) -> ((State, mdp::Action), Vec<Transition>) {
    (
        (from_state, mdp::Action(action as usize)),
        vec![(1.0, to_state, reward)],
    )
}

#[allow(clippy::needless_range_loop)]
fn print_grid(grid: &[[Cell; ROWS]; COLS]) {
    for y in 0..ROWS {
        for x in 0..COLS {
            print!("{:?}\t", grid[x][y]);
        }
        println!();
    }
}
