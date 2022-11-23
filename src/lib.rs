use crate::{
    games::{Game, SupportedGames},
    stream::VideoStream,
};
use clap::Parser;
use opencv::core::{get_num_threads, set_num_threads};
use std::thread::available_parallelism;

extern crate pretty_env_logger;
#[macro_use]
extern crate log;

pub mod games;
mod img;
pub mod pipeline;
mod roi;
mod state;
pub mod stream;

/// Stream Processor
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
pub struct Config {
    /// The URL of the stream.
    #[clap(short, long)]
    pub url: String,

    /// Show processed frames in a window
    #[clap(short, long)]
    pub show_frames: bool,

    #[clap(long, default_value_t = 0)]
    pub num_opencv_threads: i32,

    #[clap(long, default_value_t = 1)]
    pub num_libav_threads: i32,

    #[clap(long, default_value_t = 2)]
    pub process_frame_rate: i64,
}

pub async fn start(game: SupportedGames) {
    pretty_env_logger::init_timed();

    let mut config = Config::parse();

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

    let pipe = crate::pipeline::new();
    if config.show_frames {
        pipe.start_preview_thread();
    }

    let game = crate::games::new(game);
    let regions = game.regions();

    for i in 0..config.num_opencv_threads / 2 {
        pipe.process_thread(i as i32, config.show_frames, regions.clone())
            .await;
    }

    let decoder_sender = pipe.get_decode_sender();
    let mut stream = VideoStream::new(config, decoder_sender);
    stream.decode().await
}
