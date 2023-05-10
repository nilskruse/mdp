use std::collections::{BTreeMap, HashMap, HashSet};

use rand_chacha::ChaCha20Rng;

use crate::{
    mdp::{self, Action, Mdp, Reward, State},
    policies::epsilon_greedy_policy,
};

use super::StateActionAlgorithm;

pub struct MonteCarlo {
    epsilon: f64,
    max_steps: usize,
}

impl MonteCarlo {
    pub fn new(epsilon: f64, max_steps: usize) -> MonteCarlo {
        MonteCarlo { max_steps, epsilon }
    }

    fn generate_episode(
        &self,
        mdp: &Mdp,
        q_map: &BTreeMap<(State, Action), Reward>,
        rng: &mut ChaCha20Rng,
    ) -> Vec<(State, Action, Reward)> {
        let mut episode = vec![];

        let mut current_state = mdp.initial_state;
        let mut steps = 0;
        while !mdp.terminal_states.contains(&current_state) && steps < self.max_steps {
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

impl StateActionAlgorithm for MonteCarlo {
    fn run_with_q_map(
        &self,
        mdp: &crate::mdp::Mdp,
        episodes: usize,
        rng: &mut rand_chacha::ChaCha20Rng,
        q_map: &mut std::collections::BTreeMap<(crate::mdp::State, crate::mdp::Action), f64>,
    ) {
        let mut returns: BTreeMap<(State, Action), Vec<f64>> = BTreeMap::new();
        for _ in 0..episodes {
            // generate episode
            let episode = self.generate_episode(mdp, q_map, rng);
            let mut g: HashMap<(State, Action), f64> = HashMap::new();
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
}
