use std::{
    collections::{BTreeMap, HashSet},
    iter,
};

use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use rand_chacha::ChaCha20Rng;

use crate::mdp::{IndexAction, IndexState, MapMdp, Transition};

// TODO: Ensure at least one terminal state is reachable
pub fn generate_random_mdp(
    n_states: usize,
    n_actions: usize,
    n_terminal_states: usize,
    (min_actions, max_actions): (usize, usize),
    (min_transitions, max_transitions): (usize, usize),
    (min_reward, max_reward): (f64, f64),
    rng: &mut ChaCha20Rng,
) -> MapMdp<IndexState, IndexAction> {
    let mut states = vec![];
    let mut actions = vec![];

    // create states
    for i in 0..n_states {
        states.push(IndexState(i));
    }
    for i in 0..n_actions {
        actions.push(IndexAction(i));
    }

    let initial_state = IndexState(0);

    let mut states_actions = vec![];

    for state in &states {
        let n_actions = rng.gen_range(min_actions..=max_actions);
        actions.choose_multiple(rng, n_actions).for_each(|action| {
            states_actions.push((*state, *action));
        });
    }

    let transitions: BTreeMap<(IndexState, IndexAction), Vec<Transition>> =
        BTreeMap::from_iter(states_actions.iter().map(|(state, action)| {
            let n_transitions = rng.gen_range(min_transitions..=max_transitions);
            let probabilities = random_probs(n_transitions, rng);
            let mut outcomes = vec![];

            for probability in probabilities {
                let reward = rng.gen_range(min_reward..=max_reward);
                let next_state = states.choose(rng).unwrap();
                outcomes.push((probability, *next_state, reward));
            }
            ((*state, *action), outcomes)
        }));

    let terminal_states_vec = states
        .iter()
        .copied()
        .filter(|state| *state != initial_state)
        .choose_multiple(rng, n_terminal_states);
    let terminal_states: HashSet<IndexState> =
        HashSet::from_iter(terminal_states_vec.iter().copied());
    let discount_factor = 1.0;

    MapMdp {
        transitions,
        terminal_states,
        initial_state,
        discount_factor,
        states_actions,
    }
}

fn random_probs(n: usize, rng: &mut ChaCha20Rng) -> Vec<f64> {
    // Generate n-1 random floats between 0 and 1
    let mut random_numbers: Vec<f64> = iter::repeat_with(|| rng.gen_range(0.0..1.0))
        .take(n - 1)
        .collect();

    // Sort the numbers and add 0 and 1
    random_numbers.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    random_numbers.insert(0, 0.0);
    random_numbers.push(1.0);

    // Calculate the differences between consecutive numbers
    random_numbers
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect()
}
