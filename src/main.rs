extern crate pretty_env_logger;
#[macro_use]
extern crate log;

use clap::Parser;
use opencv::core::{get_num_threads, set_num_threads};
use std::thread::available_parallelism;
use stream_cv::{games::Game, stream::VideoStream};

#[tokio::main]
async fn main() {
    pretty_env_logger::init_timed();

    let mut config = stream_cv::Config::parse();

    if config.num_opencv_threads == 0 {
        let total_threads = available_parallelism().unwrap().get() as i32 / 2;
        config.num_opencv_threads = total_threads;
    }

    if !opencv::core::use_optimized().expect("error checking for optimized code") {
        debug!("changing opencv to use optimized code");
        opencv::core::set_use_optimized(true).expect("error enabling optimized code");
    } else {
        debug!("opencv is using optimized code")
    }

    set_num_threads(config.num_opencv_threads / 2).unwrap();

    let opencv_threads = get_num_threads().expect("error retrieving number of threads");
    debug!("opencv is using {} threads", opencv_threads);

    let apex = stream_cv::games::new(stream_cv::games::SupportedGames::Apex);

    let regions = apex.regions();

    let pipe = stream_cv::pipeline::new();
    if config.show_frames {
        pipe.start_preview_thread();
    }

    for i in 0..config.num_opencv_threads / 2 {
        pipe.process_thread(i as i32, config.show_frames, regions.clone())
            .await;
    }

    let decoder_sender = pipe.get_decode_sender();
    let mut stream = VideoStream::new(config, decoder_sender);
    stream.decode().await
}
