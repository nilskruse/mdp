use std::collections::BTreeMap;

use iced::alignment::Horizontal;
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
use rand_chacha::ChaCha20Rng;

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
    mdp: MAIntersectionRunnerSingleAgentRL<QLearning>,
    q_map_1: BTreeMap<(State, LightAction), f64>,
    q_map_2: BTreeMap<(State, LightAction), f64>,
    state: State,
    steps: usize,
    total_reward: f64,
    rng: ChaCha20Rng,
    cache: Cache,
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
            0.1, 0.6, 0.1, 0.6, 10, q_algo_1, q_algo_2, max_steps,
        );

        let (mut q_map_1, mut q_map_2) = mdp.gen_q_maps();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

        println!("Learning...");
        mdp.run(0, &mut q_map_1, &mut q_map_2, &mut rng);
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
                rng,
                cache: Default::default(),
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
                let (next_state, reward) =
                    self.mdp
                        .single_step(self.state, &self.q_map_1, &self.q_map_2, &mut self.rng);
                self.state = next_state;
                self.steps += 1;
                self.total_reward += reward;
                self.cache.clear();
            }
        }

        Command::none()
    }

    fn view(&self) -> Element<Message> {
        let canvas = canvas(self as &Self)
            .width(Length::Fill)
            .height(Length::Fill);

        let some_text = text(format!("state: {:?}", self.state)).size(24);
        let steps = text(format!("steps: {:?}", self.steps));
        let content = column![some_text, steps, canvas];

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(20)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(100)).map(|_| {
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
        let vis = self.cache.draw(renderer, bounds.size(), |frame| {
            let size = frame.width().min(frame.height());
            println!("size: {size}");
            let center = frame.center();
            let origin_offset = size / 2.0;
            let origin = Point::ORIGIN;
            let street_horizontal_start = Point::new(origin.x - origin_offset, origin.y);
            let street_horizontal_end = Point::new(origin.x + origin_offset, origin.y);
            let street_horizontal = Path::line(street_horizontal_start, street_horizontal_end);
            let street_width = origin_offset / 5_f32;

            let street_vertical_1_start = Point::new(
                origin.x - origin_offset / 2.0,
                origin.y - origin_offset / 2.0,
            );
            let street_vertical_1_end = Point::new(
                origin.x - origin_offset / 2.0,
                origin.y + origin_offset / 2.0,
            );
            let street_vertical_1 = Path::line(street_vertical_1_start, street_vertical_1_end);

            let street_vertical_2_start = Point::new(
                origin.x + origin_offset / 2.0,
                origin.y - origin_offset / 2.0,
            );
            let street_vertical_2_end = Point::new(
                origin.x + origin_offset / 2.0,
                origin.y + origin_offset / 2.0,
            );
            let street_vertical_2 = Path::line(street_vertical_2_start, street_vertical_2_end);
            let street_stroke = || -> Stroke {
                Stroke {
                    width: street_width,
                    style: stroke::Style::Solid(Color::BLACK),
                    ..Stroke::default()
                }
            };
            frame.translate(Vector::new(center.x, center.y));

            frame.with_save(|frame| {
                frame.stroke(&street_horizontal, street_stroke());
                frame.stroke(&street_vertical_1, street_stroke());
                frame.stroke(&street_vertical_2, street_stroke());
            });
        });

        vec![vis]
    }
}
