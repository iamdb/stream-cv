use std::{cmp::Reverse, collections::BinaryHeap, sync::Arc, thread};

use chrono::Utc;
use flume::{bounded, r#async::RecvStream, unbounded, Receiver, Sender};
use futures::StreamExt;
use opencv::{
    core::{Rect_, Scalar, ToInputOutputArray, UMat},
    dnn,
    highgui::{imshow, poll_key},
    imgproc::{put_text, rectangle, FONT_HERSHEY_SIMPLEX},
};
use tokio::{
    select, spawn,
    sync::{Mutex, MutexGuard},
};

use crate::{
    img::{self, frame::Frame},
    roi::{self, RegionOfInterestList},
    Config,
};

#[derive(Clone)]
struct OrderedFrames {
    frames: BinaryHeap<Reverse<Frame>>,
    next_frame_num: i64,
}

type SyncOrderedFrames = Arc<Mutex<OrderedFrames>>;

#[allow(dead_code)]
#[derive(Clone)]
pub struct Pipeline {
    decode_receiver: Receiver<Frame>,
    decode_sender: Sender<Frame>,
    output_receiver: Receiver<Frame>,
    output_sender: Sender<Frame>,
    preview_receiver: Receiver<Frame>,
    preview_sender: Sender<Frame>,
    process_receiver: Receiver<Frame>,
    process_sender: Sender<Frame>,
    sorted_frames: SyncOrderedFrames,
    regions: RegionOfInterestList,
    recognizer: Arc<Mutex<dnn::TextRecognitionModel>>,
    config: Config,
}

pub fn new(regions: RegionOfInterestList, config: Config) -> Pipeline {
    let count = thread::available_parallelism().unwrap().get();

    let (process_sender, process_receiver) = bounded::<Frame>(count);
    let (output_sender, output_receiver) = unbounded::<Frame>();
    let (decode_sender, decode_receiver) = bounded::<Frame>(count);
    let (preview_sender, preview_receiver) = unbounded::<Frame>();

    let sorted_frames = Arc::new(Mutex::new(OrderedFrames {
        frames: BinaryHeap::new(),
        next_frame_num: 0,
    }));

    let base_recognizer = img::make_text_recognizer();

    let recognizer = Arc::new(Mutex::new(base_recognizer));

    Pipeline {
        config,
        decode_receiver,
        decode_sender,
        output_receiver,
        output_sender,
        preview_receiver,
        preview_sender,
        process_receiver,
        process_sender,
        recognizer,
        regions,
        sorted_frames,
    }
}

#[allow(dead_code)]
impl Pipeline {
    async fn process_frame(&self, frame: Frame) -> Frame {
        if !self.regions.is_empty() {
            self.process_regions(frame.clone()).await;
        }

        // let thresh = frame
        //     .clone()
        //     .bilateral_filter(3, 220., 120.)
        //     .await
        //     .adjust_brightness(-10.0)
        //     .await
        //     .adjust_contrast(1.25)
        //     .await
        //     .threshold()
        //     .await;
        //
        // frame.add_weighted(thresh.processed_mat, 1., 0.5, 0.0).await

        frame
    }

    async fn process_regions(&self, frame: Frame) {
        for region in self.regions.iter() {
            match region.roi_type {
                roi::RegionOfInterestType::Text => {
                    frame
                        .clone()
                        .text_recognition(region.clone(), &self.recognizer)
                        .await;
                }
            }
        }
    }

    pub async fn spawn_process_thread(&self, thread_num: i32) {
        let mut stream = self.process_stream();
        loop {
            select! {
                frame = stream.next() => {
                    if let Some(f) = frame {
                        let processed_frame = self.process_frame(f).await;

                        debug!("frame {}\tprocessed\tbuffer {}\tthread_num {}", processed_frame.num, stream.len(), thread_num);
                        self.output_send(processed_frame).await;
                    }
                }
            }
        }
    }

    fn spawn_preview_thread(&self) {
        while let Ok(frame) = self.preview_receiver.recv() {
            show_frame(frame);
        }
    }

    fn process_stream(&self) -> RecvStream<Frame> {
        self.process_receiver.stream()
    }

    pub async fn process_send(&self, frame: Frame) {
        self.process_sender.send_async(frame).await.unwrap();
    }

    fn output_stream(&self) -> RecvStream<Frame> {
        self.output_receiver.stream()
    }

    pub async fn output_send(&self, frame: Frame) {
        self.output_sender.send_async(frame).await.unwrap();
    }

    fn decode_stream(&self) -> RecvStream<Frame> {
        self.decode_receiver.stream()
    }

    pub async fn decode_send(&self, frame: Frame) {
        self.decode_sender.send_async(frame).await.unwrap();
    }

    pub async fn start_router(&self) {
        let mut count = thread::available_parallelism().unwrap().get() as f64;
        let p = self.clone();

        if count > 10. {
            count -= 6.;
        }

        thread::spawn(move || {
            p.spawn_preview_thread();
        });

        for i in 0..(count - 2.) as i32 {
            let p = self.clone();
            spawn(async move {
                p.spawn_process_thread(i as i32).await;
            });
        }

        for _ in 0..2 {
            let p = self.clone();
            spawn(async move {
                p.route_frames().await;
            });
        }
    }

    async fn route_frames(&self) {
        let mut decode_stream = self.decode_stream();
        let mut output_stream = self.output_stream();

        loop {
            select! {
                frame = decode_stream.next() => {
                    let f = frame.unwrap();
                    debug!("frame {}\tdecoded\t\tbuffer {}", f.num, decode_stream.len());
                    if !f.result_text.is_empty() {
                        debug!("frame {}\ttext: {:?}", f.num, f.result_text);
                    }
                    self.process_send(f).await;
                },
                frame = output_stream.next() => {
                    let mut f = frame.unwrap();
                    let mut sf = self.sorted_frames.lock().await;

                    f.end_date = Some(Utc::now());

                    sf.frames.push(Reverse(f));

                    self.output_frame(sf);
                },

            }
        }
    }

    fn output_frame(&self, mut sf: MutexGuard<OrderedFrames>) {
        if !sf.frames.is_empty() {
            if let Some(min_frame) = sf.clone().frames.peek() {
                if min_frame.0.num == sf.next_frame_num {
                    sf.frames.pop();

                    let duration = min_frame.0.end_date.unwrap() - min_frame.0.start_date;

                    info!(
                        "frame {}\toutput\t\tduration {}\tbuffer {}",
                        min_frame.0.num,
                        format!("{}ms", duration.num_milliseconds()),
                        sf.frames.len(),
                    );

                    if self.config.show_frames {
                        self.preview_sender.send(min_frame.0.clone()).unwrap();
                    }
                    sf.next_frame_num += 2;

                    if !sf.frames.is_empty() {
                        self.output_frame(sf);
                    }
                }
            }
        }
    }
}

fn highlight_regions(mut mat: UMat, regions: Arc<Mutex<RegionOfInterestList>>) -> UMat {
    for region in regions.blocking_lock().iter() {
        let rect = Rect_::new(region.x, region.y, region.width, region.height);

        rectangle(
            &mut mat.input_output_array().unwrap(),
            rect,
            Scalar::new(0., 255., 0.0, 1.0),
            1,
            0,
            0,
        )
        .unwrap();

        put_text(
            &mut mat.input_output_array().unwrap(),
            region.result_text.as_ref().unwrap(),
            opencv::core::Point_ {
                x: region.x + region.width + 5,
                y: region.y + region.height,
            },
            FONT_HERSHEY_SIMPLEX,
            0.75,
            Scalar::new(0., 255., 0.0, 1.0),
            2,
            0,
            false,
        )
        .unwrap();
    }

    mat
}

fn show_frame(frame: Frame) {
    debug!("frame {}\tshowing frame", frame.num);

    let mat = highlight_regions(frame.processed_mat, frame.results);

    imshow("frames", &mat).unwrap();
    poll_key().unwrap();
}
