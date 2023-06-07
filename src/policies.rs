use crate::mdp::GenericMdp;
use std::collections::BTreeMap;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::mdp::{GenericAction, GenericState, IndexAction, IndexMdp, IndexState, MapMdp, Reward};

trait Policy {
    fn select_action(
        &self,
        mdp: &IndexMdp,
        q_map: &BTreeMap<(IndexState, IndexAction), Reward>,
        rng: &mut ChaCha20Rng,
    );
}

pub fn epsilon_greedy_policy<
    M: GenericMdp<S, A>,
    S: GenericState,
    A: GenericAction,
    R: Rng + SeedableRng,
>(
    mdp: &M,
    q_map: &BTreeMap<(S, A), Reward>,
    current_state: S,
    epsilon: f64,
    rng: &mut R,
) -> Option<A> {
    let random_value = rng.gen_range(0.0..1.0);
    if random_value < (1.0 - epsilon) {
        greedy_policy(mdp, q_map, current_state, rng)
    } else {
        random_policy(mdp, current_state, rng)
    }
}

pub fn greedy_policy<
    M: GenericMdp<S, A>,
    S: GenericState,
    A: GenericAction,
    R: Rng + SeedableRng,
>(
    mdp: &M,
    q_map: &BTreeMap<(S, A), Reward>,
    current_state: S,
    rng: &mut R,
) -> Option<A> {
    mdp.get_possible_actions(current_state)
        .iter()
        // extract (action, q)- tuples
        .map(|a| {
            let q = *q_map
                .get(&(current_state, *a))
                .expect("(state, action) has no q_entry");
            (*a, q)
        })
        // select action with maxium reward
        .fold(None, |prev, (current_action, current_reward)| {
            if let Some((_, prev_reward)) = prev {
                if current_reward > prev_reward {
                    Some((current_action, current_reward))
                } else if current_reward < prev_reward {
                    prev
                } else {
                    None
                }
            } else {
                Some((current_action, current_reward))
            }
        })
        // extract Option<Action> from Option<Action, Reward>
        .unzip()
        .0
        // If single max exist return that, if not select action randomly
        .or_else(|| random_policy(mdp, current_state, rng))
}

pub fn random_policy<
    M: GenericMdp<S, A>,
    S: GenericState,
    A: GenericAction,
    R: Rng + SeedableRng,
>(
    mdp: &M,
    current_state: S,
    rng: &mut R,
) -> Option<A> {
    let possible_actions = mdp.get_possible_actions(current_state);

    if possible_actions.is_empty() {
        None
    } else {
        let selected_index = rng.gen_range(0..possible_actions.len());
        Some(possible_actions[selected_index])
    }
}
