use flume::{r#async::RecvStream, unbounded, Receiver, RecvError, Sender};

use crate::frame::Frame;

#[derive(Clone)]
pub struct Pipeline {
    receiver: Receiver<Frame>,
    sender: Sender<Frame>,
}

pub fn new() -> Pipeline {
    let (sender, receiver) = unbounded::<Frame>();

    Pipeline { sender, receiver }
}

impl Pipeline {
    pub async fn recv(&self) -> Result<Frame, RecvError> {
        self.receiver.recv_async().await
    }

    pub async fn send(&self, frame: Frame) {
        self.sender.send_async(frame).await.unwrap();
    }

    pub fn stream(&self) -> RecvStream<Frame> {
        self.receiver.stream()
    }
}
