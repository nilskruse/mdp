use std::collections::BTreeMap;

use iced::executor;
use iced::mouse;
use iced::widget::canvas::{stroke, Cache, Geometry, Path, Stroke};
use iced::widget::column;

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
            0.6, 0.6, 0.6, 0.6, 10, q_algo_1, q_algo_2, max_steps,
        );

        let (mut q_map_1, mut q_map_2) = mdp.gen_q_maps();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

        println!("Learning...");
        mdp.run(100, &mut q_map_1, &mut q_map_2, &mut rng);
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

            let car_stroke = || -> Stroke {
                Stroke {
                    width: street_width / 5.0,
                    style: stroke::Style::Solid(Color::WHITE),
                    ..Stroke::default()
                }
            };

            let street_center_1 = Point::new(origin.x - origin_offset / 2.0, origin.y);
            let street_center_2 = Point::new(origin.x + origin_offset / 2.0, origin.y);
            let street_frac = 3.0;
            let car_length = street_width / 5.0;
            frame.translate(Vector::new(center.x, center.y));

            frame.with_save(|frame| {
                frame.stroke(&street_horizontal, street_stroke());
                frame.stroke(&street_vertical_1, street_stroke());
                frame.stroke(&street_vertical_2, street_stroke());
            });

            frame.with_save(|frame| {
                let mut draw_vertical_cars = |street_center: Point, ns_cars: u8| {
                    let top_street = Point::new(
                        street_center.x - street_width / 3.5,
                        street_center.y - street_width / street_frac,
                    );

                    let bot_street = Point::new(
                        street_center.x + street_width / 3.5,
                        street_center.y + street_width / street_frac,
                    );

                    for car in 1..=ns_cars {
                        let cars = car / 2 + car % 2;
                        let car_line = if car % 2 == 1 {
                            // car rendered on top
                            let car_1 = Point::new(
                                top_street.x,
                                top_street.y - cars as f32 * car_length * 2.0,
                            );
                            let car_2 = Point::new(
                                top_street.x,
                                top_street.y - cars as f32 * car_length * 2.0 + car_length,
                            );
                            Path::line(car_1, car_2)
                        } else {
                            // car rendered on bottom
                            let car_1 = Point::new(
                                bot_street.x,
                                bot_street.y + cars as f32 * car_length * 2.0,
                            );
                            let car_2 = Point::new(
                                bot_street.x,
                                bot_street.y + cars as f32 * car_length * 2.0 - car_length,
                            );
                            Path::line(car_1, car_2)
                        };

                        frame.stroke(&car_line, car_stroke());
                    }
                };
                if self.state.ns_cars_1 > 0 {
                    draw_vertical_cars(street_center_1, self.state.ns_cars_1);
                }
                if self.state.ns_cars_2 > 0 {
                    draw_vertical_cars(street_center_2, self.state.ns_cars_2);
                }
            });

            frame.with_save(|frame| {
                let mut draw_horizontal_cars = |street_center: Point, ns_cars: u8| {
                    let left_street = Point::new(
                        street_center.x - street_width / street_frac,
                        street_center.y + street_width / 3.5,
                    );

                    let right_street = Point::new(
                        street_center.x + street_width / street_frac,
                        street_center.y - street_width / 3.5,
                    );

                    for car in 1..=ns_cars {
                        let cars = car / 2 + car % 2;
                        let car_line = if car % 2 == 1 {
                            // car rendered on top
                            let car_1 = Point::new(
                                left_street.x - cars as f32 * car_length * 2.0,
                                left_street.y,
                            );
                            let car_2 = Point::new(
                                left_street.x - cars as f32 * car_length * 2.0 + car_length,
                                left_street.y,
                            );
                            Path::line(car_1, car_2)
                        } else {
                            // car rendered on bottom
                            let car_1 = Point::new(
                                right_street.x + cars as f32 * car_length * 2.0,
                                right_street.y,
                            );
                            let car_2 = Point::new(
                                right_street.x + cars as f32 * car_length * 2.0 - car_length,
                                right_street.y,
                            );
                            Path::line(car_1, car_2)
                        };

                        frame.stroke(&car_line, car_stroke());
                    }
                };
                if self.state.ew_cars_1 > 0 {
                    draw_horizontal_cars(street_center_1, self.state.ew_cars_1);
                }
                if self.state.ew_cars_2 > 0 {
                    draw_horizontal_cars(street_center_2, self.state.ew_cars_2);
                }
            });
        });

        vec![vis]
    }
}
