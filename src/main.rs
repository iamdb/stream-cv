extern crate pretty_env_logger;
#[macro_use]
extern crate log;

mod frame;
mod pipeline;
mod stream;

use frame::Frame;
use std::env;
use stream::VideoStream;

#[tokio::main]
async fn main() {
    pretty_env_logger::init_timed();
    let url = &env::args().nth(1).expect("cannot open");
    let pipe = pipeline::new();

    pipeline::start_router(pipe.clone()).await;

    let mut stream = VideoStream::new(url.to_string(), &pipe);
    stream.decode().await
}
