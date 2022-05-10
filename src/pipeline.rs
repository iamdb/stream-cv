use flume::{r#async::RecvStream, unbounded, Receiver, Sender};

use crate::frame::Frame;

#[allow(dead_code)]
#[derive(Clone)]
pub struct Pipeline {
    decode_receiver: Receiver<Frame>,
    decode_sender: Sender<Frame>,
    output_receiver: Receiver<Frame>,
    output_sender: Sender<Frame>,
    process_receiver: Receiver<Frame>,
    process_sender: Sender<Frame>,
}

pub fn new() -> Pipeline {
    let (process_sender, process_receiver) = unbounded::<Frame>();
    let (output_sender, output_receiver) = unbounded::<Frame>();
    let (decode_sender, decode_receiver) = unbounded::<Frame>();

    Pipeline {
        decode_receiver,
        decode_sender,
        output_receiver,
        output_sender,
        process_receiver,
        process_sender,
    }
}

impl Pipeline {
    pub fn process_stream(&self) -> RecvStream<Frame> {
        self.process_receiver.stream()
    }

    pub async fn process_send(&self, frame: Frame) {
        self.process_sender.send_async(frame).await.unwrap();
    }
    pub fn output_stream(&self) -> RecvStream<Frame> {
        self.output_receiver.stream()
    }

    pub async fn output_send(&self, frame: Frame) {
        self.output_sender.send_async(frame).await.unwrap();
    }

    pub fn decode_stream(&self) -> RecvStream<Frame> {
        self.decode_receiver.stream()
    }

    pub async fn decode_send(&self, frame: Frame) {
        self.decode_sender.send_async(frame).await.unwrap();
    }
}
