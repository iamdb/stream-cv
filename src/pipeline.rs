use chrono::Utc;
use flume::{bounded, Receiver, Sender};
use futures::StreamExt;
use opencv::{
    dnn::{self, TextRecognitionModel},
    highgui::{imshow, poll_key},
};
use std::{sync::Arc, thread};
use tokio::{select, sync::Mutex};

use crate::{
    games::apex::Weapon,
    img::{self, frame::Frame},
    roi::RegionOfInterestList,
    state::{self, GameState},
    Config,
};

#[allow(dead_code)]
#[derive(Clone)]
pub struct Pipeline<'f> {
    decode_receiver: Receiver<Frame>,
    decode_sender: Sender<Frame>,
    preview_receiver: Receiver<Frame>,
    preview_sender: Sender<Frame>,
    process_receiver: Receiver<&'f mut Frame>,
    process_sender: Sender<&'f mut Frame>,
    regions: RegionOfInterestList,
    recognizer: Arc<Mutex<dnn::TextRecognitionModel>>,
    config: Config,
    state: GameState,
}

pub fn new<'f>(regions: RegionOfInterestList, config: Config) -> Pipeline<'f> {
    let (process_sender, process_receiver) = bounded::<&mut Frame>(60);
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

impl<'f> Pipeline<'f> {
    pub fn get_decode_sender(&self) -> Sender<Frame> {
        self.decode_sender.clone()
    }
    pub async fn start_router(&self) {
        if self.config.show_frames {
            let p = self.preview_receiver.clone();
            thread::spawn(move || {
                spawn_preview_thread(p);
            });
        }

        let preview_send = self.preview_sender.clone();
        let decode_receiver = self.decode_receiver.clone();
        let recognizer = self.recognizer.clone();
        let show_frames = self.config.show_frames;
        let regions = self.regions.clone();

        tokio::spawn(async move {
            let mut decode_stream = decode_receiver.stream();

            loop {
                select! {
                    frame = decode_stream.next() => {
                        if let Some(mut f) = frame {
                           debug!("frame {}\tdecoded\t\tbuffer {}", f.num, decode_stream.len());

                            let start_time = Utc::now();
                            process_frame(&mut f, regions.clone(), &recognizer).await;
                            let process_time = Utc::now() - start_time;

                            debug!("frame {}\tprocessed\tduration {}ms", f.num, process_time.num_milliseconds());
                            f.end_date = Some(Utc::now());

                            // let w1 = f.results.get_value("weapon_1_name".to_string()).unwrap();
                            // let w2 = f.results.get_value("weapon_2_name".to_string()).unwrap();
                            // let sim1 = Weapon::match_string(w1.result.as_ref().unwrap().to_string());
                            // let sim2 = Weapon::match_string(w2.result.as_ref().unwrap().to_string());
                            //
                            // println!("::::::::::::::::::::::::::::::::: sim1: {:?} sim2: {:?}", sim1, sim2);

                            info!("frame {}\toutput", f.num);

                            if show_frames {
                                preview_send.send(f.clone()).unwrap();
                            }
                        }
                    }
                }
            }
        });
    }
}

fn show_frame(mut frame: Frame) {
    debug!("frame {}\tshowing frame", frame.num);

    frame.highlight_regions();

    imshow("frames", &frame.processed_mat).unwrap();
    poll_key().unwrap();
}

fn spawn_preview_thread(recv: Receiver<Frame>) {
    while let Ok(frame) = recv.recv() {
        show_frame(frame);
    }
}

async fn process_frame<'f>(
    frame: &'f mut Frame,
    regions: RegionOfInterestList,
    recognizer: &Arc<Mutex<TextRecognitionModel>>,
) {
    frame
        .bilateral_filter(9, 75., 75.)
        .await
        .adjust_contrast(1.55)
        .await
        .adjust_brightness(-10.)
        .await;

    if regions.is_empty() {
        process_regions(frame, regions, recognizer).await;
    }
}

async fn process_regions<'f>(
    frame: &'f mut Frame,
    regions: RegionOfInterestList,
    recognizer: &Arc<Mutex<TextRecognitionModel>>,
) {
    frame.list_text_recognition(regions, recognizer).await;
}
