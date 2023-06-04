use crate::{
    envs::cliff_walking::{self, Action, CLIFF_REWARD, COLS, ROWS, STEP_REWARD},
    mdp::{self, MapMdp, Mdp, Reward, State, Transition},
};

use super::cliff_walking::{CliffWalkingAction, CliffWalkingState, CliffWalkingTransition};

// cliff walking with a slippy cliff that pushes down the agent with a certain probability
pub fn build_mdp(slip_prob: f64) -> Mdp {
    // build regular cliff walking mdp
    let mut mdp = cliff_walking::build_mdp();

    // change transitions near cliff to add slip possibility

    let row = ROWS - 2; // row above cliff

    // all states above clip
    for col in 1..COLS - 1 {
        let from_state = State(row * COLS + col);

        // Left action
        let action = Action::Left;
        let to_state = State(row * COLS + (col - 1)); // state we want to go to
        let slip_state = State((row + 1) * COLS + col); // state we might slip in

        let (key, value) =
            build_slippy_transition(from_state, to_state, slip_state, action, slip_prob);
        mdp.transitions.entry(key).and_modify(|e| *e = value);

        // Right action
        let action = Action::Right;
        let to_state = State(row * COLS + (col + 1)); // state we want to go to
        let slip_state = State((row + 1) * COLS + col); // state we might slip in

        let (key, value) =
            build_slippy_transition(from_state, to_state, slip_state, action, slip_prob);
        mdp.transitions.entry(key).and_modify(|e| *e = value);
    }

    mdp
}

fn build_slippy_transition(
    from_state: State,
    to_state: State,
    slip_state: State,
    action: Action,
    slip_prob: f64,
) -> ((State, mdp::Action), Vec<Transition>) {
    (
        (from_state, mdp::Action(action as usize)),
        vec![
            (1.0 - slip_prob, to_state, STEP_REWARD),
            (slip_prob, slip_state, CLIFF_REWARD),
        ],
    )
}

pub fn build_generic_mdp(
    slip_prob: f64,
) -> anyhow::Result<MapMdp<CliffWalkingState, CliffWalkingAction>> {
    let mut mdp = cliff_walking::build_generic_mdp()?;

    let row = ROWS - 2; // row above cliff

    // all states above clip
    for col in 1..COLS - 1 {
        let from_state = CliffWalkingState(row, col);

        // Left action
        let action = CliffWalkingAction::Left;
        let to_state = CliffWalkingState(row, col - 1); // state we want to go to
        let slip_state = CliffWalkingState(row + 1, col); // state we might slip in

        let (key, value) =
            build_slippy_transition_generic(from_state, to_state, slip_state, action, slip_prob);
        mdp.transitions.entry(key).and_modify(|e| *e = value);

        // Right action
        let action = CliffWalkingAction::Right;
        let to_state = CliffWalkingState(row, col + 1); // state we want to go to
        let slip_state = CliffWalkingState(row + 1, col); // state we might slip in

        let (key, value) =
            build_slippy_transition_generic(from_state, to_state, slip_state, action, slip_prob);
        mdp.transitions.entry(key).and_modify(|e| *e = value);
    }

    Ok(mdp)
}

fn build_slippy_transition_generic(
    from_state: CliffWalkingState,
    to_state: CliffWalkingState,
    slip_state: CliffWalkingState,
    action: CliffWalkingAction,
    slip_prob: f64,
) -> (
    (CliffWalkingState, CliffWalkingAction),
    Vec<CliffWalkingTransition>,
) {
    (
        (from_state, action),
        vec![
            (1.0 - slip_prob, to_state, STEP_REWARD),
            (slip_prob, slip_state, CLIFF_REWARD),
        ],
    )
}
