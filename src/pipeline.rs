use chrono::Utc;
use flume::{bounded, r#async::RecvStream, Receiver, Sender};
use futures::StreamExt;
use opencv::{
    dnn,
    highgui::{imshow, poll_key},
};
use std::{sync::Arc, thread};
use tokio::{select, spawn, sync::Mutex};

use crate::{
    games::apex::Weapon,
    img::{self, frame::Frame},
    roi::RegionOfInterestList,
    state::{self, GameState},
    Config,
};

#[allow(dead_code)]
#[derive(Clone)]
pub struct Pipeline {
    decode_receiver: Receiver<Frame>,
    decode_sender: Sender<Frame>,
    preview_receiver: Receiver<Frame>,
    preview_sender: Sender<Frame>,
    process_receiver: Receiver<Frame>,
    process_sender: Sender<Frame>,
    regions: RegionOfInterestList,
    recognizer: Arc<Mutex<dnn::TextRecognitionModel>>,
    config: Config,
    state: GameState,
}

pub fn new(regions: RegionOfInterestList, config: Config) -> Pipeline {
    let (process_sender, process_receiver) = bounded::<Frame>(60);
    let (decode_sender, decode_receiver) = bounded::<Frame>(60);
    let (preview_sender, preview_receiver) = bounded::<Frame>(60);

    let base_recognizer = img::make_text_recognizer();

    let recognizer = Arc::new(Mutex::new(base_recognizer));

    Pipeline {
        config,
        decode_receiver,
        decode_sender,
        preview_receiver,
        preview_sender,
        process_receiver,
        process_sender,
        recognizer,
        regions,
        state: state::new(),
    }
}

#[allow(dead_code)]
impl Pipeline {
    async fn process_frame(&self, mut frame: Frame) -> Frame {
        frame
            .bilateral_filter(9, 75., 75.)
            .await
            .adjust_contrast(1.55)
            .await
            .adjust_brightness(-10.)
            .await;

        if !self.regions.is_empty() {
            frame = self.process_regions(frame).await;
        }

        frame
    }

    async fn process_regions(&self, frame: Frame) -> Frame {
        frame
            .list_text_recognition(self.regions.clone(), &self.recognizer)
            .await
    }

    fn process_stream(&self) -> RecvStream<Frame> {
        self.process_receiver.stream()
    }

    pub async fn process_send(&self, frame: Frame) {
        self.process_sender.send_async(frame).await.unwrap();
    }

    fn decode_stream(&self) -> RecvStream<Frame> {
        self.decode_receiver.stream()
    }

    pub async fn decode_send(&self, frame: Frame) {
        self.decode_sender.send_async(frame).await.unwrap();
    }

    pub async fn start_router(&self) {
        if self.config.show_frames {
            let p = self.clone();
            thread::spawn(move || {
                p.spawn_preview_thread();
            });
        }

        let p = self.clone();
        spawn(async move {
            p.route_frames().await;
        });
    }

    fn spawn_preview_thread(&self) {
        while let Ok(frame) = self.preview_receiver.recv() {
            show_frame(frame);
        }
    }

    async fn route_frames(&self) {
        let mut decode_stream = self.decode_stream();
        let mut process_stream = self.process_stream();

        loop {
            select! {
                frame = decode_stream.next() => {
                    if let Some(f) = frame {
                       debug!("frame {}\tdecoded\t\tbuffer {}", f.num, decode_stream.len());
                       self.process_send(f).await;
                    }
                }
                frame = process_stream.next() => {
                    if let Some(f) = frame {
                        let start_time = Utc::now();
                        let mut processed_frame = self.process_frame(f).await;
                        let process_time = Utc::now() - start_time;

                        debug!("frame {}\tprocessed\tduration {}ms\tbuffer {}", processed_frame.num, process_time.num_milliseconds(), self.process_receiver.len());
                        processed_frame.end_date = Some(Utc::now());

                        let w1 = processed_frame.results.get_value("weapon_1_name".to_string()).unwrap();
                        let w2 = processed_frame.results.get_value("weapon_2_name".to_string()).unwrap();
                        let sim1 = Weapon::match_string(w1.result.as_ref().unwrap().to_string());
                        let sim2 = Weapon::match_string(w2.result.as_ref().unwrap().to_string());

                        println!("::::::::::::::::::::::::::::::::: sim1: {:?} sim2: {:?}", sim1, sim2);

                        self.output_frame(processed_frame);
                    }
                }
            }
        }
    }

    fn output_frame(&self, frame: Frame) {
        info!("frame {}\toutput", frame.num,);

        if self.config.show_frames {
            self.preview_sender.send(frame).unwrap();
        }
    }

    pub fn get_decode_sender(self) -> Sender<Frame> {
        self.decode_sender
    }
}

fn show_frame(mut frame: Frame) {
    debug!("frame {}\tshowing frame", frame.num);

    frame.highlight_regions();

    imshow("frames", &frame.processed_mat).unwrap();
    poll_key().unwrap();
}
