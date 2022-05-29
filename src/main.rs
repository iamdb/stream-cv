extern crate pretty_env_logger;
#[macro_use]
extern crate log;

mod img;
mod pipeline;
mod roi;
mod stream;

use std::thread::available_parallelism;

use clap::Parser;
use img::frame::Frame;
use opencv::core::{get_num_threads, set_num_threads};
use stream::VideoStream;

use self::roi::StreamResolution;

/// Stream Processor
#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]
pub struct Config {
    /// The URL of the stream.
    #[clap(short, long)]
    url: String,

    /// Show processed frames in a window
    #[clap(short, long)]
    show_frames: bool,

    #[clap(long, default_value_t = 0)]
    num_opencv_threads: i32,

    #[clap(long, default_value_t = 1)]
    num_libav_threads: i32,

    #[clap(long, default_value_t = 5)]
    process_frame_rate: i64,
}

#[tokio::main]
async fn main() {
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

    set_num_threads(config.num_opencv_threads).unwrap();

    let opencv_threads = get_num_threads().expect("error retrieving number of threads");
    debug!("opencv is using {} threads", opencv_threads);

    let regions = make_regions();

    let pipe = pipeline::new(regions, config.clone());
    pipe.start_router().await;

    let decoder_sender = pipe.get_decode_sender();
    let mut stream = VideoStream::new(config, decoder_sender);
    stream.decode().await
}

fn make_regions() -> roi::RegionOfInterestList {
    let mut list = roi::new_region_list();

    list.add_new_region(
        "loaded_mag_size".to_string(),
        1720,
        960,
        62,
        40,
        StreamResolution::HD1080p,
    );
    list.add_new_region(
        "total_ammo".to_string(),
        1720,
        998,
        62,
        30,
        StreamResolution::HD1080p,
    );
    list.add_new_region(
        "weapon_1_name".to_string(),
        1555,
        1034,
        110,
        24,
        StreamResolution::HD1080p,
    );
    list.add_new_region(
        "weapon_2_name".to_string(),
        1715,
        1034,
        110,
        24,
        StreamResolution::HD1080p,
    );
    list.add_new_region(
        "compass_number".to_string(),
        935,
        90,
        50,
        32,
        StreamResolution::HD1080p,
    );

    list
}
