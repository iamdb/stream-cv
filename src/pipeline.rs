use flume::{unbounded, Receiver, RecvError, Sender};

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
    pub async fn process_recv(&self) -> Result<Frame, RecvError> {
        self.process_receiver.recv_async().await
    }

    pub async fn process_send(&self, frame: Frame) {
        self.process_sender.send_async(frame).await.unwrap();
    }

    pub async fn output_recv(&self) -> Result<Frame, RecvError> {
        self.output_receiver.recv_async().await
    }

    pub async fn output_send(&self, frame: Frame) {
        self.output_sender.send_async(frame).await.unwrap();
    }

    pub async fn decode_recv(&self) -> Result<Frame, RecvError> {
        self.decode_receiver.recv_async().await
    }

    pub async fn decode_send(&self, frame: Frame) {
        self.decode_sender.send_async(frame).await.unwrap();
    }
}
