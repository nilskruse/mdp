use std::collections::{BTreeMap, HashSet};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{
    mdp::{
        self, GenericAction, GenericMdp, GenericState, IndexAction, IndexMdp, IndexState, Reward,
    },
    policies::epsilon_greedy_policy,
};

use super::GenericStateActionAlgorithm;

pub struct MonteCarlo {
    epsilon: f64,
    max_steps: usize,
}

impl MonteCarlo {
    pub fn new(epsilon: f64, max_steps: usize) -> MonteCarlo {
        MonteCarlo { max_steps, epsilon }
    }

    fn generate_episode<
        M: GenericMdp<S, A>,
        S: GenericState,
        A: GenericAction,
        R: Rng + SeedableRng,
    >(
        &self,
        mdp: &M,
        q_map: &BTreeMap<(S, A), Reward>,
        rng: &mut R,
    ) -> Vec<(S, A, Reward)> {
        let mut episode = vec![];

        let mut current_state = mdp.get_initial_state(rng);
        let mut steps = 0;
        while !mdp.is_terminal(current_state) && steps < self.max_steps {
            let selected_action =
                epsilon_greedy_policy(mdp, q_map, current_state, self.epsilon, rng);

            let Some(selected_action) = selected_action else {break};
            let (next_state, reward) = mdp.perform_action((current_state, selected_action), rng);

            episode.push((current_state, selected_action, reward));
            current_state = next_state;
            steps += 1;
        }
        episode
    }
}

impl GenericStateActionAlgorithm for MonteCarlo {
    fn run_with_q_map<
        M: GenericMdp<S, A>,
        S: GenericState,
        A: GenericAction,
        R: Rng + SeedableRng,
    >(
        &self,
        mdp: &M,
        episodes: usize,
        rng: &mut R,
        q_map: &mut BTreeMap<(S, A), f64>,
    ) {
        let mut returns: BTreeMap<(S, A), Vec<f64>> = BTreeMap::new();
        for _ in 0..episodes {
            // generate episode
            let episode = self.generate_episode(mdp, q_map, rng);
            let mut g: BTreeMap<(S, A), f64> = BTreeMap::new();
            let mut visited_states = HashSet::new();

            // get first-visit returns
            episode.iter().for_each(|(state, action, reward)| {
                g.entry((*state, *action))
                    .and_modify(|e| *e += *reward)
                    .or_insert(*reward);
                // track all visited states for later while we're at it
                visited_states.insert(state);
            });

            // append to returns(s, a)
            g.iter().for_each(|((state, action), ret)| {
                returns
                    .entry((*state, *action))
                    .and_modify(|v| v.push(*ret))
                    .or_insert(vec![*ret]);
            });

            // average into q_map
            returns.iter().for_each(|(key, rets)| {
                q_map
                    .entry(*key)
                    .and_modify(|entry| *entry = rets.iter().sum::<f64>() / rets.len() as f64);
            });

            returns.clear();
        }
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }
}
