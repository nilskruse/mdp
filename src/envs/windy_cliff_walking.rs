use std::collections::HashMap;

use crate::{
    envs::cliff_walking::{Action, Cell, CLIFF_REWARD, COLS, END_REWARD, ROWS, STEP_REWARD},
    mdp::{self, Mdp, State, Transition},
};

// cliff walking with a downward wind that pushes down the agent with a certain probability
pub fn build_mdp() -> Mdp {
    let mut grid: [[Cell; ROWS]; COLS] = [[Cell::Regular; ROWS]; COLS];

    //set lower row
    grid[0][ROWS - 1] = Cell::Start;
    for cell in grid.iter_mut().take(COLS - 1).skip(1) {
        cell[ROWS - 1] = Cell::Cliff;
    }
    grid[COLS - 1][ROWS - 1] = Cell::End;

    let mut transitions: HashMap<(State, mdp::Action), Vec<Transition>> = HashMap::new();

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

    for col in 1..COLS {
        terminal_states.push(State((ROWS - 1) * COLS + col));
    }

    Mdp {
        transitions,
        terminal_states,
        initial_state: State((ROWS - 1) * COLS),
    }
}

fn build_transition(
    from_state_1: State,
    from_state_2: State,
    action: Action,
    step_reward: f64,
) -> ((State, mdp::Action), Vec<Transition>) {
    todo!()
}
