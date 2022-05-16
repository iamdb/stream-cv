use std::{cmp::Reverse, collections::BinaryHeap, sync::Arc, thread};

use flume::{bounded, r#async::RecvStream, Receiver, Sender};
use futures::StreamExt;
use opencv::highgui::{imshow, poll_key};
use tokio::{
    select, spawn,
    sync::{Mutex, MutexGuard},
};

use crate::frame::Frame;

#[derive(Debug, Clone)]
struct OrderedFrames {
    frames: BinaryHeap<Reverse<Frame>>,
    next_frame_num: i64,
}

type SyncOrderedFrames = Arc<Mutex<OrderedFrames>>;

#[derive(Clone)]
pub struct Pipeline {
    decode_receiver: Receiver<Frame>,
    decode_sender: Sender<Frame>,
    output_receiver: Receiver<Frame>,
    output_sender: Sender<Frame>,
    process_receiver: Receiver<Frame>,
    process_sender: Sender<Frame>,
    sorted_frames: SyncOrderedFrames,
}

pub fn new() -> Pipeline {
    let (process_sender, process_receiver) = bounded::<Frame>(120);
    let (output_sender, output_receiver) = bounded::<Frame>(60);
    let (decode_sender, decode_receiver) = bounded::<Frame>(120);

    let sorted_frames = Arc::new(Mutex::new(OrderedFrames {
        frames: BinaryHeap::new(),
        next_frame_num: 0,
    }));

    Pipeline {
        decode_receiver,
        decode_sender,
        output_receiver,
        output_sender,
        process_receiver,
        process_sender,
        sorted_frames,
    }
}

impl Pipeline {
    async fn process_frame(&self, frame: Frame) -> Frame {
        frame
            .bilateral_filter()
            .await
            //.detail_enhance()
            //.await
            .dilate()
            .await
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
        let count = thread::available_parallelism().unwrap().get();

        for i in 0..(count - 2) {
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
                    self.process_send(f).await;
                },
                frame = output_stream.next() => {
                    let f = frame.unwrap();
                    let mut sf = self.sorted_frames.lock().await;
                    let next_frame_num = sf.next_frame_num + 1;

                    sf.frames.push(Reverse(f));

                    self.output_frame(sf, next_frame_num, output_stream.len() as i64);
                },

            }
        }
    }

    fn output_frame(
        &self,
        mut sf: MutexGuard<OrderedFrames>,
        next_frame_num: i64,
        stream_len: i64,
    ) {
        if !sf.frames.is_empty() {
            if let Some(min_frame) = sf.clone().frames.peek() {
                if min_frame.0.num == sf.next_frame_num {
                    sf.frames.pop();
                    debug!(
                        "frame {}\toutput\t\tnext_frame {}\tbuffer {}",
                        min_frame.0.num, next_frame_num, stream_len
                    );
                    sf.next_frame_num += 1;

                    //show_frame(min_frame);
                }
            }
        }
    }

    fn _show_frame(frame: &Reverse<Frame>) {
        imshow("frames", &frame.0.processed_mat).unwrap();
        poll_key().unwrap();
    }
}
