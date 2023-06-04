use rand::{Rng, SeedableRng};

use crate::mdp::GenericMdp;

pub struct BlackjackMdp;

impl BlackjackMdp {
    pub fn new() -> Self {
        Self {}
    }

    pub fn draw_card<R: Rng + SeedableRng>(&self, rng: &mut R) -> u8 {
        rng.gen_range(0..=10)
    }
}

//BlackjackState = (current_sum: 12-21, dealer card: 1-10, usable ace: true/false)
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub enum BlackjackState {
    Running(u8, u8, bool),
    Win,
    Draw,
    Loss,
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone, Hash, Copy)]
pub enum BlackjackAction {
    Hit = 0,
    Stick = 1,
}

impl GenericMdp<BlackjackState, BlackjackAction> for BlackjackMdp {
    fn add_transition_vector(
        &mut self,
        sa: (BlackjackState, BlackjackAction),
        transition: Vec<(crate::mdp::Probability, BlackjackState, crate::mdp::Reward)>,
    ) -> anyhow::Result<()> {
        todo!()
    }

    fn add_terminal_state(&mut self, state: BlackjackState) {
        todo!()
    }

    fn perform_action<R: rand::SeedableRng + rand::Rng>(
        &self,
        state_action: (BlackjackState, BlackjackAction),
        rng: &mut R,
    ) -> (BlackjackState, crate::mdp::Reward) {
        let card = self.draw_card(rng);

        let (state, action) = state_action;
        match state {
            BlackjackState::Running(player, dealer, ace) => todo!(),
            BlackjackState::Win => todo!(),
            BlackjackState::Draw => todo!(),
            BlackjackState::Loss => todo!(),
        }
        todo!()
    }

    fn get_possible_actions(&self, _: BlackjackState) -> Vec<BlackjackAction> {
        vec![BlackjackAction::Hit, BlackjackAction::Stick]
    }

    fn get_all_state_actions_iter(
        &self,
    ) -> std::collections::btree_map::Keys<
        '_,
        (BlackjackState, BlackjackAction),
        Vec<(crate::mdp::Probability, BlackjackState, crate::mdp::Reward)>,
    > {
        todo!()
    }

    fn is_terminal(&self, state: BlackjackState) -> bool {
        match state {
            BlackjackState::Running(_, _, _) => false,
            BlackjackState::Win => true,
            BlackjackState::Draw => true,
            BlackjackState::Loss => true,
        }
    }

    fn get_initial_state(&self) -> BlackjackState {
        todo!()
    }

    fn get_discount_factor(&self) -> f64 {
        1.0
    }
}
