extern crate pretty_env_logger;
#[macro_use]
extern crate log;

mod img;
mod pipeline;
mod roi;
mod stream;

use img::frame::Frame;
use libvips::VipsApp;
use opencv::core::{get_num_threads, set_num_threads};
use std::env;
use stream::VideoStream;

#[tokio::main]
async fn main() {
    pretty_env_logger::init_timed();
    let url = &env::args().nth(1).expect("cannot open");

    if !opencv::core::use_optimized().expect("error checking for optimized code") {
        debug!("changing opencv to use optimized code");
        opencv::core::set_use_optimized(true).expect("error enabling optimized code");
    } else {
        debug!("opencv is using optimized code")
    }

    set_num_threads(2).unwrap();

    let opencv_threads = get_num_threads().expect("error retrieving number of threads");
    debug!("opencv is using {} threads", opencv_threads);

    let app = VipsApp::new("stream-cv", false).expect("error creating vips app");
    app.concurrency_set(2);

    let regions = make_regions();

    let pipe = pipeline::new(regions);
    pipe.start_router().await;

    let mut stream = VideoStream::new(url.to_string(), &pipe);
    stream.decode().await
}

fn make_regions() -> roi::RegionOfInterestList {
    let mut list = roi::new_region_list();

    list.add_new_region(1509, 53, 300, 150);

    list
}
