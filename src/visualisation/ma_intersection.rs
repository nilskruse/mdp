use std::collections::BTreeMap;

use iced::executor;
use iced::mouse;
use iced::widget::canvas::{stroke, Cache, Geometry, LineCap, Path, Stroke};
use iced::widget::column;
use iced::widget::row;
use iced::widget::text;
use iced::widget::{canvas, container};
use iced::{
    Application, Color, Command, Element, Length, Point, Rectangle, Renderer, Settings,
    Subscription, Theme, Vector,
};
use rand::SeedableRng;

use crate::algorithms::q_learning::QLearning;
use crate::algorithms::GenericStateActionAlgorithm;
use crate::mdp::GenericMdp;
use crate::mdp::Reward;
use crate::multiagent::intersection::{LightAction, MAIntersectionRunnerSingleAgentRL, State};

pub fn main() -> iced::Result {
    MAIntersection::run(Settings {
        antialiasing: true,
        ..Settings::default()
    })
}

struct MAIntersection {
    // struct MAIntersection {
    mdp: MAIntersectionRunnerSingleAgentRL<QLearning>,
    q_map_1: BTreeMap<(State, LightAction), f64>,
    q_map_2: BTreeMap<(State, LightAction), f64>,
    state: State,
    steps: usize,
    total_reward: f64,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Tick(time::OffsetDateTime),
}

impl Application for MAIntersection {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let max_steps = 5000;
        let q_algo_1 = QLearning::new(0.1, 0.1, max_steps);
        let q_algo_2 = QLearning::new(0.1, 0.1, max_steps);

        let mdp = MAIntersectionRunnerSingleAgentRL::new(
            0.2, 0.2, 0.2, 0.2, 10, q_algo_1, q_algo_2, max_steps,
        );

        let (mut q_map_1, mut q_map_2) = mdp.gen_q_maps();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

        println!("Learning...");

        mdp.run(1000, &mut q_map_1, &mut q_map_2, &mut rng);
        let state = mdp.mdp.get_initial_state(&mut rng);

        println!("Done!");

        (
            MAIntersection {
                mdp,
                q_map_1,
                q_map_2,
                state,
                steps: 0,
                total_reward: 0.0,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("Clock - Iced")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Tick(_) => {
                let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
                let (next_state, reward) =
                    self.mdp
                        .single_step(self.state, &self.q_map_1, &self.q_map_2, &mut rng);
                self.state = next_state;
                self.steps += 1;
                self.total_reward += reward;
            }
        }

        Command::none()
    }

    fn view(&self) -> Element<Message> {
        let canvas = canvas(self as &Self)
            .width(Length::Fill)
            .height(Length::Fill);

        let some_text = text(format!("{:?}", self.state));
        let content = column![some_text, canvas];

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(20)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(1)).map(|_| {
            Message::Tick(
                time::OffsetDateTime::now_local()
                    .unwrap_or_else(|_| time::OffsetDateTime::now_utc()),
            )
        })
    }
}

impl<Message> canvas::Program<Message, Renderer> for MAIntersection {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        vec![]
    }
}
