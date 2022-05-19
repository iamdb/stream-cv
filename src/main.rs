extern crate pretty_env_logger;
#[macro_use]
extern crate log;

mod frame;
mod pipeline;
mod stream;

use frame::Frame;
use libvips::VipsApp;
use std::env;
use stream::VideoStream;

#[tokio::main]
async fn main() {
    pretty_env_logger::init_timed();
    let url = &env::args().nth(1).expect("cannot open");

    let app = VipsApp::new("stream-cv", false).unwrap();
    app.concurrency_set(2);

    let pipe = pipeline::new();
    pipe.start_router().await;

    let mut stream = VideoStream::new(url.to_string(), &pipe);
    stream.decode().await
}
