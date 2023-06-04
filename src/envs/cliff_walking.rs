use crate::mdp::GenericMdp;
use std::collections::BTreeMap;

use crate::mdp::{
    self, GenericState, IndexMdp, IndexState, MapMdp, Probability, Reward, Transition,
};

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

#[allow(clippy::needless_range_loop)]
fn print_grid(grid: &[[Cell; ROWS]; COLS]) {
    for y in 0..ROWS {
        for x in 0..COLS {
            print!("{:?}\t", grid[x][y]);
        }
        println!();
    }
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub struct CliffWalkingState(pub usize, pub usize);

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub enum CliffWalkingAction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

pub type CliffWalkingTransition = (Probability, CliffWalkingState, Reward);

pub fn build_mdp() -> anyhow::Result<MapMdp<CliffWalkingState, CliffWalkingAction>> {
    // create grid of cells
    let mut grid: [[Cell; ROWS]; COLS] = [[Cell::Regular; ROWS]; COLS];

    //set lower row
    grid[0][ROWS - 1] = Cell::Start;
    for cell in grid.iter_mut().take(COLS - 1).skip(1) {
        cell[ROWS - 1] = Cell::Cliff;
    }
    grid[COLS - 1][ROWS - 1] = Cell::End;

    let mut mdp =
        MapMdp::<CliffWalkingState, CliffWalkingAction>::new(CliffWalkingState(ROWS - 1, 0));

    // fill transition map for every state and action
    for row in 0..ROWS {
        for col in 0..COLS {
            let from_state = CliffWalkingState(row, col);

            // Up action
            // can only run into wall with up
            let action = CliffWalkingAction::Up;
            let (key, value) = if row as isize - 1 < 0 {
                build_transition(from_state, from_state, action, STEP_REWARD)
            } else {
                let to_state = CliffWalkingState(row - 1, col);
                build_transition(from_state, to_state, action, STEP_REWARD)
            };
            mdp.add_transition_vector(key, value)?;

            // Left action
            // can only run into wall with left
            let action = CliffWalkingAction::Left;
            let (key, value) = if col as isize - 1 < 0 {
                build_transition(from_state, from_state, action, STEP_REWARD)
            } else {
                let to_state = CliffWalkingState(row, col - 1);
                build_transition(from_state, to_state, action, STEP_REWARD)
            };
            mdp.add_transition_vector(key, value)?;

            //Right action
            let action = CliffWalkingAction::Right;
            let (key, value) = if col + 1 == COLS {
                build_transition(from_state, from_state, action, STEP_REWARD)
            } else {
                let to_state = CliffWalkingState(row, col + 1);
                match grid[col + 1][row] {
                    Cell::Start => build_transition(from_state, to_state, action, STEP_REWARD),
                    Cell::Regular => build_transition(from_state, to_state, action, STEP_REWARD),
                    Cell::Cliff => build_transition(from_state, to_state, action, CLIFF_REWARD),
                    Cell::End => build_transition(from_state, to_state, action, END_REWARD),
                }
            };
            mdp.add_transition_vector(key, value)?;

            //Down action
            let action = CliffWalkingAction::Down;
            let (key, value) = if row + 1 == ROWS {
                build_transition(from_state, from_state, action, STEP_REWARD)
            } else {
                let to_state = CliffWalkingState(row + 1, col);
                match grid[col][row + 1] {
                    Cell::Start => build_transition(from_state, to_state, action, STEP_REWARD),
                    Cell::Regular => build_transition(from_state, to_state, action, STEP_REWARD),
                    Cell::Cliff => build_transition(from_state, to_state, action, CLIFF_REWARD),
                    Cell::End => build_transition(from_state, to_state, action, END_REWARD),
                }
            };
            mdp.add_transition_vector(key, value)?;
        }
    }

    // last row all terminal except first cell
    for col in 1..COLS {
        mdp.add_terminal_state(CliffWalkingState(ROWS - 1, col));
    }

    // print_grid(&grid);

    Ok(mdp)
}

fn build_transition(
    from_state: CliffWalkingState,
    to_state: CliffWalkingState,
    action: CliffWalkingAction,
    reward: Reward,
) -> (
    (CliffWalkingState, CliffWalkingAction),
    Vec<CliffWalkingTransition>,
) {
    ((from_state, action), vec![(1.0, to_state, reward)])
}
